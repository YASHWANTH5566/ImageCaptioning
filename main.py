import os
import string
import random
import numpy as np
from pickle import dump, load
from PIL import Image
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tqdm import tqdm
import tensorflow as tf

# ---------------------------
# 1) Load and clean captions
# ---------------------------
def load_doc(filename):
    with open(filename, 'r') as file:
        text = file.read()
    return text

def all_img_captions(filename):
    file = load_doc(filename)
    captions = file.split('\n')
    descriptions = {}
    for caption in captions[:-1]:
        if not caption.strip():
            continue
        img, cap = caption.split(',', 1)
        if img not in descriptions:
            descriptions[img] = [cap]
        else:
            descriptions[img].append(cap)
    return descriptions

def cleaning_text(captions):
    table = str.maketrans('', '', string.punctuation)
    for img, caps in captions.items():
        for i, cap in enumerate(caps):
            cap = cap.replace("-", " ")
            desc = cap.split()
            desc = [word.lower() for word in desc]
            desc = [word.translate(table) for word in desc]
            desc = [word for word in desc if len(word) > 1 and word.isalpha()]
            captions[img][i] = ' '.join(desc)
    return captions

def save_descriptions(descriptions, filename):
    lines = []
    for key, desc_list in descriptions.items():
        for desc in desc_list:
            lines.append(f"{key}\t{desc}")
    with open(filename, 'w') as f:
        f.write("\n".join(lines))

# ---------------------------
# 2) Extract image features (run once; skip if features.pkl exists)
# ---------------------------
def extract_features(directory, out_path="features.pkl"):
    model = Xception(include_top=False, pooling='avg', weights='imagenet')
    features = {}
    valid_images = ['.jpg', '.jpeg', '.png']
    for img in tqdm(os.listdir(directory), desc="Extracting features"):
        ext = os.path.splitext(img)[1].lower()
        if ext not in valid_images:
            continue
        filename = os.path.join(directory, img)
        image = Image.open(filename).convert("RGB")
        image = image.resize((299, 299))
        image = np.array(image).astype('float32')
        image = np.expand_dims(image, axis=0)
        image = image / 127.5
        image = image - 1.0
        feat = model.predict(image, verbose=0)
        features[img] = feat
    dump(features, open(out_path, "wb"))
    return features

# ---------------------------
# 3) Tokenizer and max length
# ---------------------------
def dict_to_list(descriptions):
    all_desc = []
    for key in descriptions.keys():
        [all_desc.append(d) for d in descriptions[key]]
    return all_desc

def create_tokenizer(descriptions):
    desc_list = dict_to_list(descriptions)
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(desc_list)
    return tokenizer

def max_length(descriptions):
    return max(len(d.split()) for d in dict_to_list(descriptions))

# ---------------------------
# 4) Prepare training data
# ---------------------------
def load_photos(filename, images_folder):
    file = load_doc(filename)
    photos = file.replace('\r', '').split("\n")
    photos = [p.strip() for p in photos if p.strip()]
    photos_present = [photo for photo in photos if os.path.exists(os.path.join(images_folder, photo))]
    missing = len(photos) - len(photos_present)
    if missing:
        print(f"  Warning: {missing} listed images not found in {images_folder}")
    return photos_present

def load_clean_descriptions(filename, photos):
    file = load_doc(filename)
    descriptions = {}
    photos_set = set(photos)
    for line in file.split("\n"):
        if len(line.strip()) == 0:
            continue
        img, desc = line.split('\t')
        if img in photos_set:
            if img not in descriptions:
                descriptions[img] = []
            descriptions[img].append(f"startseq {desc} endseq")
    return descriptions

def load_features_for_photos(photos, features_path="features.pkl"):
    all_features = load(open(features_path, "rb"))
    return {k: all_features[k] for k in photos if k in all_features}

# ---------------------------
# 5) Batched, shuffled data generator
# ---------------------------
def create_sequences(tokenizer, max_len, desc_list, feature, vocab_size):
    X1, X2, y = [], [], []
    for desc in desc_list:
        seq = tokenizer.texts_to_sequences([desc])[0]
        for i in range(1, len(seq)):
            in_seq, out_seq = seq[:i], seq[i]
            in_seq = pad_sequences([in_seq], maxlen=max_len)[0]
            X1.append(feature)
            X2.append(in_seq)
            y.append(out_seq)  # keep as integer; one-hot later per-batch to save memory
    return X1, X2, y

def data_generator(descriptions, features, tokenizer, max_len, vocab_size, batch_size=64, shuffle=True):
    keys = list(descriptions.keys())
    while True:
        if shuffle:
            random.shuffle(keys)

        X1_batch, X2_batch, y_batch = [], [], []
        for key in keys:
            if key not in features:
                continue
            feature = features[key][0]
            desc_list = descriptions[key]
            X1, X2, y = create_sequences(tokenizer, max_len, desc_list, feature, vocab_size)
            X1_batch.extend(X1)
            X2_batch.extend(X2)
            y_batch.extend(y)

            while len(X1_batch) >= batch_size:
                bx1 = np.array(X1_batch[:batch_size])
                bx2 = np.array(X2_batch[:batch_size])
                by = to_categorical(y_batch[:batch_size], num_classes=vocab_size)
                yield (bx1, bx2), by
                X1_batch = X1_batch[batch_size:]
                X2_batch = X2_batch[batch_size:]
                y_batch = y_batch[batch_size:]

        # yield any remainder before looping to next epoch
        if X1_batch:
            bx1 = np.array(X1_batch)
            bx2 = np.array(X2_batch)
            by = to_categorical(y_batch, num_classes=vocab_size)
            yield (bx1, bx2), by

def count_total_sequences(descriptions):
    total = 0
    for desc_list in descriptions.values():
        for d in desc_list:
            total += max(len(d.split()) - 1, 0)
    return total

# ---------------------------
# 6) Define model — stronger image/text fusion via concatenate
#    instead of add(), plus a slightly deeper decoder.
# ---------------------------
def define_model(vocab_size, max_len):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_len,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    # concatenate instead of add: preserves image signal instead of
    # blending it away into the same 256-dim space as the LSTM state.
    decoder0 = concatenate([fe2, se3])
    decoder1 = Dense(256, activation='relu')(decoder0)
    decoder1 = Dropout(0.3)(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    model.summary()
    return model

# ---------------------------
# 7) TF Dataset wrapper (batched)
# ---------------------------
def create_tf_dataset(descriptions, features, tokenizer, max_len, vocab_size, batch_size, shuffle=True):
    def gen():
        yield from data_generator(descriptions, features, tokenizer, max_len, vocab_size,
                                    batch_size=batch_size, shuffle=shuffle)

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            (
                tf.TensorSpec(shape=(None, 2048), dtype=tf.float32),
                tf.TensorSpec(shape=(None, max_len), dtype=tf.int32),
            ),
            tf.TensorSpec(shape=(None, vocab_size), dtype=tf.float32),
        )
    )

# ---------------------------
# 8) Main execution
# ---------------------------
if __name__ == "__main__":
    root = "Filckr8k_Dataset"
    images_folder = os.path.join(root, "Images")
    train_file = os.path.join(root, "Flickr_8k.trainImages.txt")
    dev_file = os.path.join(root, "Flickr_8k.devImages.txt")

    batch_size = 64
    epochs = 30  # early stopping will likely halt before this

    print("Loading train/dev images...")
    train_imgs = load_photos(train_file, images_folder)
    dev_imgs = load_photos(dev_file, images_folder)

    train_descriptions = load_clean_descriptions("descriptions.txt", train_imgs)
    dev_descriptions = load_clean_descriptions("descriptions.txt", dev_imgs)

    train_features = load_features_for_photos(train_imgs)
    dev_features = load_features_for_photos(dev_imgs)

    print(f"  Train images with features: {len(train_features)}")
    print(f"  Dev images with features:   {len(dev_features)}")

    print("Loading tokenizer...")
    tokenizer = load(open("tokenizer.p", "rb"))
    vocab_size = len(tokenizer.word_index) + 1

    # IMPORTANT: compute max_length over the SAME corpus the tokenizer
    # was fit on (train captions), and reuse it consistently everywhere,
    # including at inference time in test.py.
    max_len = max_length(train_descriptions)
    print(f"Vocab size: {vocab_size}, max_len: {max_len}")

    print("Defining model...")
    model = define_model(vocab_size, max_len)

    train_steps = count_total_sequences(train_descriptions) // batch_size
    dev_steps = max(count_total_sequences(dev_descriptions) // batch_size, 1)

    os.makedirs("models2", exist_ok=True)

    train_dataset = create_tf_dataset(train_descriptions, train_features, tokenizer,
                                       max_len, vocab_size, batch_size, shuffle=True)
    train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    dev_dataset = create_tf_dataset(dev_descriptions, dev_features, tokenizer,
                                     max_len, vocab_size, batch_size, shuffle=False)
    dev_dataset = dev_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True, verbose=1),
        ModelCheckpoint(
            filepath=os.path.join("models2", "best_model.h5"),
            monitor='val_loss',
            save_best_only=True,
            verbose=1,
        ),
    ]

    print("Training model...")
    history = model.fit(
        train_dataset,
        steps_per_epoch=train_steps,
        validation_data=dev_dataset,
        validation_steps=dev_steps,
        epochs=epochs,
        callbacks=callbacks,
        verbose=1,
    )

    model.save(os.path.join("models2", "final_model.h5"))
    print("Done. Use models2/best_model.h5 (lowest val_loss) for inference, not the last epoch.")