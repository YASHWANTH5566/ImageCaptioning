"""
Image Caption Generator — Gradio web app for Hugging Face Spaces.

Architecture and decoding logic mirror test.py exactly (same model
definition, same beam search with n-gram repeat blocking and end-of-
sequence bias) so behavior matches what was validated locally.
"""

import os
import numpy as np
from pickle import load
from PIL import Image

import gradio as gr
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.layers import Input, Dense, LSTM, Embedding, Dropout, concatenate
from tensorflow.keras.models import Model

# -----------------------
# Config — these files must be uploaded alongside this app.py in the Space
# -----------------------
TOKENIZER_PATH = "tokenizer.p"
MODEL_WEIGHTS_PATH = "best_model.h5"
MAX_LENGTH = 35  # must match max_len printed during training in main.py

START_TOKEN = "startseq"
END_TOKEN = "endseq"

# -----------------------
# Model definition — MUST match main.py's define_model exactly,
# layer-for-layer, or load_weights will throw a shape mismatch.
# -----------------------
def define_model(vocab_size, max_length):
    inputs1 = Input(shape=(2048,), name='input_1')
    fe1 = Dropout(0.3)(inputs1)
    fe2 = Dense(256, activation='relu')(fe1)

    inputs2 = Input(shape=(max_length,), name='input_2')
    se1 = Embedding(vocab_size, 256, mask_zero=True)(inputs2)
    se2 = Dropout(0.3)(se1)
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
def extract_features(pil_image, xception_model):
    image = pil_image.convert("RGB")
    image = image.resize((299, 299))
    image = np.array(image).astype('float32')
    image = np.expand_dims(image, axis=0)
    image = image / 127.5
    image = image - 1.0
    feature = xception_model.predict(image, verbose=0)
    return feature

# -----------------------
# Repeat-blocking helper
# -----------------------
def _violates_repeat_rules(words, block_repeat_ngram, max_consecutive_word_repeats):
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
# Beam search decoding with length normalization, repeat blocking,
# and end-of-sequence probability bias (see test.py for rationale).
# -----------------------
def generate_caption_beam(model, tokenizer, index_word, photo, max_length,
                           beam_width=5, block_repeat_ngram=3,
                           max_consecutive_word_repeats=1, length_penalty=0.7,
                           max_words=14, end_bias_start=8, end_bias_strength=0.15):
    start_seq = tokenizer.texts_to_sequences([START_TOKEN])[0]
    sequences = [(start_seq, 0.0)]
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

            top_indices = np.argsort(preds)[-(beam_width * 4):]

            seq_words = [index_word.get(t) for t in seq]
            added = 0
            for idx in top_indices[::-1]:
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

        def normalized_score(item):
            seq, score = item
            length = max(len(seq), 1)
            return score / (length ** length_penalty)

        ordered = sorted(all_candidates, key=normalized_score, reverse=True)
        sequences = ordered[:beam_width]

        if all(index_word.get(s[-1]) == END_TOKEN for s, _ in sequences):
            break

    best_seq = sorted(
        sequences,
        key=lambda item: item[1] / (max(len(item[0]), 1) ** length_penalty),
        reverse=True
    )[0][0]
    words = [index_word.get(idx) for idx in best_seq if index_word.get(idx) is not None]
    return ' '.join(words)

def clean_caption(text):
    words = text.split()
    words = [w for w in words if w not in (START_TOKEN, END_TOKEN)]
    return ' '.join(words)

# -----------------------
# Load everything ONCE at startup (not per-request) — this is the
# single biggest factor in keeping response times reasonable on a
# free CPU Space.
# -----------------------
print("Loading tokenizer...")
tokenizer = load(open(TOKENIZER_PATH, "rb"))
vocab_size = len(tokenizer.word_index) + 1
index_word = {index: word for word, index in tokenizer.word_index.items()}

print("Loading caption model...")
caption_model = define_model(vocab_size, MAX_LENGTH)
caption_model.load_weights(MODEL_WEIGHTS_PATH)

print("Loading Xception feature extractor...")
xception_model = Xception(include_top=False, pooling="avg", weights="imagenet")

print("Models loaded. Ready to serve.")

# -----------------------
# Gradio inference function
# -----------------------
def predict_caption(image):
    if image is None:
        return "Please upload an image."
    try:
        photo = extract_features(image, xception_model)
        raw_caption = generate_caption_beam(
            caption_model, tokenizer, index_word, photo, MAX_LENGTH, beam_width=5
        )
        return clean_caption(raw_caption)
    except Exception as e:
        return f"Error generating caption: {e}"

# -----------------------
# Gradio UI
# -----------------------
demo = gr.Interface(
    fn=predict_caption,
    inputs=gr.Image(label="Upload an image", type="pil", sources=["upload", "webcam", "clipboard"]),
    outputs=gr.Textbox(label="Generated Caption"),
    title="Image Caption Generator",
    description=(
        "Upload a photo and get an AI-generated caption describing it. "
        "Built with a Xception CNN for image features and an LSTM decoder "
        "trained on the Flickr8k dataset."
    ),
    flagging_mode="never",
)

if __name__ == "__main__":
    demo.launch()
