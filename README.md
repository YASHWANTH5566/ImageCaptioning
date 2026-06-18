# 🖼️ Image Caption Generator

Generate natural-language captions for any image using a CNN + LSTM architecture, trained from scratch on the Flickr8k dataset and deployed as a free, public web app.

**🔗 Live demo:** https://huggingface.co/spaces/Dev-73/image-caption-generator

---

## What it does

Upload any photo and get back a generated sentence describing what's happening in it — e.g. *"a dog is running through the grass with a boy chasing it."*


## How it works

| Component | Role |
|---|---|
| **Xception** (CNN, pretrained on ImageNet) | Extracts a 2048-dim feature vector from each input image |
| **Embedding + LSTM** | Encodes the partial caption generated so far |
| **Concatenation + Dense decoder** | Combines image features and language state to predict the next word |
| **Beam search** | Generates the final caption, with repetition-blocking and length normalization for cleaner output |

Trained on **Flickr8k**: 8,091 images, ~40,000 human-written captions (5 per image).

## Architecture

```
Image (299×299×3)
      │
      ▼
  Xception (frozen, ImageNet weights)
      │
      ▼
  2048-d feature vector ──────────┐
                                   │
Partial caption (token sequence)  │
      │                           │
      ▼                           │
  Embedding (256-d)                │
      │                           │
      ▼                           │
  LSTM (256 units)                 │
      │                           │
      └──────────► Concatenate ◄──┘
                       │
                       ▼
                 Dense (256, ReLU)
                       │
                       ▼
              Dense (vocab_size, softmax)
                       │
                       ▼
                  Next word
```

## Results & honest limitations

This was a learning project, and I'm upfront about where it stands:

- Early/short phrases are often accurate ("a man is riding a bike," "a dog is running").
- Captions can drift into generic filler on longer outputs, since the underlying LSTM decoder has no attention mechanism — it sees a single fixed image vector at every decoding step rather than attending to different image regions per word.
- Performance reflects Flickr8k's domain (people, pets, everyday outdoor scenes) — it generalizes poorly outside that, e.g. specialized objects, indoor scenes, or unusual compositions.

**What I'd do with more time:** add a Bahdanau/Luong-style attention mechanism over Xception's spatial feature maps (rather than a single pooled vector), and/or fine-tune on a larger, more diverse captioning dataset (COCO Captions).

## Problems I actually debugged

A non-exhaustive list of what came up in the process — kept here because the debugging mattered more than the tutorial-following:

- **Model returned the same caption for every image.** Caused by weak image/language fusion (`add()` merging both branches into the same 256-d space, letting the language model dominate) and zero validation tracking, so there was no signal for when training started memorizing instead of generalizing. Fixed with a `concatenate()`-based fusion, a real train/dev split, and early stopping.
- **Captions degenerated into repeated phrases after the first clause** (e.g. "...in the background in the background..."). Root cause was two-fold: a padding-length mismatch between training and inference that silently truncated the model's own generated context once captions grew long, plus the fact that the end-of-sequence token is the rarest training target per caption, so the model had little incentive to learn to stop confidently. Fixed via correcting the padding length, n-gram repeat blocking, length-normalized beam search, and a soft end-of-sequence probability bias.
- **Deployment dependency hell** (TensorFlow build availability, Python 3.13 stdlib changes, stale Gradio/huggingface_hub API compatibility) — each surfaced as a different runtime error during Hugging Face Spaces deployment and required reading actual build logs rather than guessing at version pins.

## Tech stack

`Python` · `TensorFlow / Keras` · `Xception` · `LSTM` · `NumPy` · `Pillow` · `Gradio` · `Hugging Face Spaces`

## Project structure

```
├── main.py                      # Training script (data loading, model definition, training loop)
├── test.py                      # Local inference/testing with beam search decoding
├── create_splits_dynamic.py     # Generates train/dev/test image splits
├── count_images.py              # Dataset sanity-check utility
├── app.py                       # Gradio web app (deployed to Hugging Face Spaces)
├── requirements.txt             # Web app dependencies
└── README.md
```

## Running locally

```bash
# 1. Clone the repo
git clone https://github.com/yourusername/image-caption-generator.git
cd image-caption-generator

# 2. Install dependencies
pip install tensorflow pillow numpy tqdm

# 3. Download Flickr8k dataset (images + captions) and place under:
#    Filckr8k_Dataset/Images/
#    Filckr8k_Dataset/Flickr_8k.trainImages.txt, .devImages.txt, .testImages.txt

# 4. Train
python main.py

# 5. Generate a caption for a single image
python test.py
```

## Running the web app locally

```bash
pip install -r requirements.txt
python app.py
```

Then open the local URL Gradio prints (typically `http://127.0.0.1:7860`).

## Dataset

[Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) — 8,091 images with 5 human-annotated captions each, commonly used as an entry-level benchmark for image captioning research.

## License

MIT — feel free to fork, learn from, or build on this.
