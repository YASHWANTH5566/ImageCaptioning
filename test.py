from tensorflow.keras.preprocessing.sequence import pad_sequences
from keras.applications.xception import Xception
from keras.models import Model
from pickle import load
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout
from tensorflow.keras.layers import concatenate
import os
import glob
import random

# Must match the tokens used in main.py's load_clean_descriptions exactly.
START_TOKEN = 'startseq'
END_TOKEN = 'endseq'

# -----------------------
# Model definition (MUST match main.py's define_model exactly,
# layer-for-layer, or load_weights will throw a shape mismatch)
# -----------------------
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.5)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.5)(se1)
    se3 = LSTM(256)(se2)

    decoder0 = concatenate([fe2, se3])
    decoder1 = Dense(256, activation='relu')(decoder0)
    decoder1 = Dropout(0.3)(decoder1)
    outputs = Dense(vocab_size, activation='softmax')(decoder1)

    model = Model(inputs=[inputs1, inputs2], outputs=outputs)
    model.compile(loss='categorical_crossentropy', optimizer='adam')
    return model

# -----------------------
# Feature extraction
# -----------------------
def extract_features(filename, model):
    try:
        image = Image.open(filename)
    except:
        raise Exception("ERROR: Couldn't open image! Check the path and extension.")

    image = image.convert("RGB")  # handles grayscale/RGBA/palette images safely
    image = image.resize((299, 299))
    image = np.array(image).astype('float32')
    image = np.expand_dims(image, axis=0)

    image = image / 127.5
    image = image - 1.0

    feature = model.predict(image, verbose=0)
    return feature

# -----------------------
# Word lookup (cache as dict instead of O(n) scan each call)
# -----------------------
def build_index_word_map(tokenizer):
    return {index: word for word, index in tokenizer.word_index.items()}

# -----------------------
# GREEDY decoding with repetition blocking
# -----------------------
def generate_desc_greedy(model, tokenizer, index_word, photo, max_length,
                          block_repeat_ngram=3, max_consecutive_word_repeats=1,
                          max_words=14, end_bias_start=8, end_bias_strength=0.15):
    in_text = START_TOKEN
    words = [START_TOKEN]
    end_id = tokenizer.word_index.get(END_TOKEN)

    for i in range(min(max_length, max_words)):
        sequence = tokenizer.texts_to_sequences([in_text])[0]
        sequence = pad_sequences([sequence], maxlen=max_length)
        preds = model.predict([photo, sequence], verbose=0)[0]

        # Once the caption has reached a reasonable length, nudge endseq's
        # probability upward a little each additional step. This makes the
        # model more willing to stop instead of relying purely on it having
        # learned a confident endseq signal on its own (which it hasn't,
        # since endseq is the rarest training target per caption).
        if end_id is not None and len(words) - 1 >= end_bias_start:
            extra_steps = (len(words) - 1) - end_bias_start
            preds = preds.copy()
            preds[end_id] += end_bias_strength * extra_steps

        ranked = np.argsort(preds)[::-1]
        chosen = None
        for idx in ranked:
            word = index_word.get(idx)
            if word is None:
                continue
            candidate_words = words + [word]
            if _violates_repeat_rules(candidate_words, block_repeat_ngram,
                                       max_consecutive_word_repeats):
                continue
            chosen = word
            break

        if chosen is None:
            break

        words.append(chosen)
        in_text += ' ' + chosen
        if chosen == END_TOKEN:
            break

    return ' '.join(words)

def _violates_repeat_rules(words, block_repeat_ngram, max_consecutive_word_repeats):
    """Returns True if appending the last word in `words` creates either:
       - more than max_consecutive_word_repeats of the same word in a row, or
       - a repeated n-gram (of size block_repeat_ngram) that already occurred.
    """
    if len(words) >= max_consecutive_word_repeats + 1:
        last_word = words[-1]
        run = 0
        for w in reversed(words):
            if w == last_word:
                run += 1
            else:
                break
        if run > max_consecutive_word_repeats:
            return True

    n = block_repeat_ngram
    if len(words) >= n:
        new_ngram = tuple(words[-n:])
        for j in range(len(words) - n):
            if tuple(words[j:j + n]) == new_ngram:
                return True

    return False

# -----------------------
# BEAM SEARCH decoding with length normalization and repeat blocking.
# Length normalization stops the search from favoring long rambling
# sequences purely because they have more tokens to accumulate score
# proportionally; n-gram blocking stops "in the background in the
# background..." style loops directly.
# -----------------------
def generate_desc_beam(model, tokenizer, index_word, photo, max_length,
                        beam_width=5, block_repeat_ngram=3,
                        max_consecutive_word_repeats=1, length_penalty=0.7,
                        max_words=14, end_bias_start=8, end_bias_strength=0.15):
    start_seq = tokenizer.texts_to_sequences([START_TOKEN])[0]
    sequences = [(start_seq, 0.0)]  # (token list, cumulative log-prob)
    end_id = tokenizer.word_index.get(END_TOKEN)

    for _ in range(min(max_length, max_words)):
        all_candidates = []
        for seq, score in sequences:
            if len(seq) > 0 and index_word.get(seq[-1]) == END_TOKEN:
                all_candidates.append((seq, score))
                continue

            padded = pad_sequences([seq], maxlen=max_length)
            preds = model.predict([photo, padded], verbose=0)[0]
            preds = np.clip(preds, 1e-12, 1.0)

            if end_id is not None and len(seq) - 1 >= end_bias_start:
                extra_steps = (len(seq) - 1) - end_bias_start
                preds = preds.copy()
                preds[end_id] += end_bias_strength * extra_steps

            # look at more candidates than beam_width so that blocked
            # (repetitive) ones still leave enough valid options
            top_indices = np.argsort(preds)[-(beam_width * 4):]

            seq_words = [index_word.get(t) for t in seq]
            added = 0
            for idx in top_indices[::-1]:  # best first
                word = index_word.get(idx)
                if word is None:
                    continue
                candidate_words = seq_words + [word]
                if _violates_repeat_rules(candidate_words, block_repeat_ngram,
                                           max_consecutive_word_repeats):
                    continue
                candidate_seq = seq + [idx]
                candidate_score = score + np.log(preds[idx])
                all_candidates.append((candidate_seq, candidate_score))
                added += 1
                if added >= beam_width:
                    break

        if not all_candidates:
            break

        # rank by length-normalized score so short, clean endings aren't
        # penalized relative to long sequences that just keep talking
        def normalized_score(item):
            seq, score = item
            length = max(len(seq), 1)
            return score / (length ** length_penalty)

        ordered = sorted(all_candidates, key=normalized_score, reverse=True)
        sequences = ordered[:beam_width]

        if all(index_word.get(s[-1]) == END_TOKEN for s, _ in sequences):
            break

    best_seq = sorted(sequences, key=lambda item: item[1] / (max(len(item[0]), 1) ** length_penalty),
                       reverse=True)[0][0]
    words = [index_word.get(idx) for idx in best_seq if index_word.get(idx) is not None]
    return ' '.join(words)

# -----------------------
# Caption cleanup
# -----------------------
def clean_caption(text):
    words = text.split()
    words = [w for w in words if w not in (START_TOKEN, END_TOKEN)]
    return ' '.join(words)

# -----------------------
# Main program
# -----------------------
if __name__ == "__main__":
    # -----------------------
    # Configurations — EDIT THESE PATHS
    # -----------------------
    root = r"C:\Users\myasw\OneDrive\Desktop\Image caption generator"
    images_folder = os.path.join(root, "Filckr8k_Dataset", "Images")
    tokenizer_path = os.path.join(root, "tokenizer.p")
    model_weights_path = os.path.join(root, "models2", "best_model.h5")
    max_length = 35  # must match max_len printed by main.py during training

    # Set to a specific filename to test one image, or None to test several random ones
    single_img_filename = "161669933_3e7d8c7e2c.jpg"
    num_random_images_if_none = 5

    # Load tokenizer
    tokenizer = load(open(tokenizer_path, "rb"))
    vocab_size = len(tokenizer.word_index) + 1
    index_word = build_index_word_map(tokenizer)

    # Define model and load weights
    model = define_model(vocab_size, max_length)
    model.load_weights(model_weights_path)

    # Load Xception model for feature extraction
    xception_model = Xception(include_top=False, pooling="avg", weights="imagenet")

    # Build list of images to test
    if single_img_filename:
        test_images = [os.path.join(images_folder, single_img_filename)]
    else:
        all_imgs = glob.glob(os.path.join(images_folder, "*.jpg"))
        test_images = random.sample(all_imgs, min(num_random_images_if_none, len(all_imgs)))

    for img_path in test_images:
        print("=" * 70)
        print("Image:", os.path.basename(img_path))

        photo = extract_features(img_path, xception_model)

        greedy = generate_desc_greedy(model, tokenizer, index_word, photo, max_length)
        beam = generate_desc_beam(model, tokenizer, index_word, photo, max_length, beam_width=5)

        print("Greedy  :", clean_caption(greedy))
        print("Beam(5) :", clean_caption(beam))

    # If testing multiple images and greedy gives the *exact same* output
    # every time but beam search gives different, more sensible captions,
    # that confirms decoding (not the model) was the main symptom.
    # If BOTH greedy and beam give near-identical captions across very
    # different images, the model itself collapsed during training and
    # needs retraining with the fixes in main.py.