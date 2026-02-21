<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python"/>
  <img src="https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white" alt="TensorFlow"/>
  <img src="https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white" alt="Keras"/>
  <img src="https://img.shields.io/badge/Streamlit-UI-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white" alt="Streamlit"/>
  <img src="https://img.shields.io/badge/License-MIT-green?style=for-the-badge" alt="License"/>
</p>

<h1 align="center">🖼️ Image Caption Generator</h1>

<p align="center">
  <em>An AI-powered deep learning application that generates natural language captions for images using a CNN–LSTM encoder-decoder architecture, trained on the Flickr8k dataset.</em>
</p>

<p align="center">
  <a href="#-key-features">Features</a> •
  <a href="#-architecture">Architecture</a> •
  <a href="#-demo">Demo</a> •
  <a href="#-quick-start">Quick Start</a> •
  <a href="#-model-details">Model Details</a> •
  <a href="#-project-structure">Project Structure</a> •
  <a href="#-results">Results</a>
</p>

---

## ✨ Key Features

| Feature | Description |
|---|---|
| 🧠 **DenseNet201 Feature Extractor** | Leverages a pre-trained DenseNet201 CNN to extract rich 1920-dimensional image embeddings |
| 📝 **LSTM Caption Decoder** | Generates fluent, human-readable captions word-by-word using sequence modelling |
| 🌐 **Streamlit Web Interface** | Clean, interactive UI — upload any image and get an AI-generated caption in seconds |
| ⚡ **End-to-End Pipeline** | From raw image → feature extraction → tokenized caption — fully automated inference |
| 📊 **Jupyter Notebook** | Complete, reproducible training pipeline with visualizations and learning curves |

---

## 🏗️ Architecture

The model follows a classic **encoder-decoder** framework widely used in image captioning research:

```
┌─────────────────────────────────────────────────────────────────────┐
│                    IMAGE CAPTION GENERATOR                         │
├─────────────────────────────────────────────────────────────────────┤
│                                                                     │
│   📷 Input Image (224×224×3)                                        │
│         │                                                           │
│         ▼                                                           │
│   ┌──────────────┐                                                  │
│   │  DenseNet201  │  ← Pre-trained on ImageNet                      │
│   │  (Encoder)    │                                                  │
│   └──────┬───────┘                                                  │
│          │                                                           │
│          ▼                                                           │
│   Image Features (1×1920)                                           │
│          │                                                           │
│          ▼                                                           │
│   Dense(256, ReLU) ──→ Reshape(1, 256)                              │
│                              │                                       │
│                    ┌─────────┴─────────┐                             │
│                    │    Concatenate     │                             │
│                    └─────────┬─────────┘                             │
│                              │                                       │
│   "startseq" → Embedding(256) → Text Embeddings                    │
│                              │                                       │
│                              ▼                                       │
│                      ┌──────────────┐                                │
│                      │  LSTM (256)   │  ← Sequential word generation │
│                      └──────┬───────┘                                │
│                             │                                        │
│                      Dropout(0.5)                                    │
│                             │                                        │
│                     Add (residual)                                   │
│                             │                                        │
│                      Dense(128, ReLU)                                │
│                      Dropout(0.5)                                    │
│                             │                                        │
│                      Dense(softmax)  → Predicted Word                │
│                             │                                        │
│                             ▼                                        │
│                  "a dog playing in a field"                           │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

**How it works:**
1. **Encoder** — DenseNet201 (pre-trained on ImageNet) extracts a 1920-dim feature vector from the input image.
2. **Decoder** — An LSTM network takes the image features concatenated with word embeddings and generates a caption one word at a time, starting from `startseq` until it predicts `endseq`.

---

## 🎬 Demo

1. Launch the Streamlit app
2. Upload any `.jpg`, `.jpeg`, or `.png` image
3. The AI generates a descriptive caption and displays it overlaid on the image

```
📷  Upload  →  🧠 DenseNet201 + LSTM  →  📝 "two dogs are playing in the grass"
```

---

## 🚀 Quick Start

### Prerequisites

- Python **3.8+**
- pip (Python package manager)

### 1. Clone the Repository

```bash
git clone https://github.com/prvn-kumar01/ImageCaptioning.git
cd ImageCaptioning
```

### 2. Install Dependencies

```bash
pip install tensorflow numpy matplotlib streamlit pillow
```

### 3. Run the Application

```bash
streamlit run main.py
```

The app will open in your browser at `http://localhost:8501`. Upload an image and see the magic! ✨

---

## 🧪 Model Details

| Attribute | Value |
|---|---|
| **Feature Extractor** | DenseNet201 (pre-trained on ImageNet) |
| **Image Embedding Size** | 1920 dimensions |
| **Word Embedding Size** | 256 dimensions |
| **LSTM Hidden Units** | 256 |
| **Max Caption Length** | 34 tokens |
| **Input Image Size** | 224 × 224 × 3 |
| **Training Dataset** | [Flickr8k](https://www.kaggle.com/datasets/adityajn105/flickr8k) (~8,000 images, 5 captions each) |
| **Train/Val Split** | 85% / 15% |
| **Regularization** | Dropout (0.5) + Residual connection |
| **Framework** | TensorFlow / Keras |

### Training Pipeline

```
Flickr8k Dataset
      │
      ├── Images ──→ DenseNet201 ──→ Feature Vectors (1920-d)
      │
      └── Captions ──→ Text Preprocessing ──→ Tokenization ──→ Padded Sequences
                                                                      │
                                                                      ▼
                                                           Custom Data Generator
                                                                      │
                                                                      ▼
                                                              CNN-LSTM Model
                                                                      │
                                                                      ▼
                                                          Trained Caption Model
```

**Text Preprocessing Steps:**
- Convert to lowercase
- Remove special characters and numbers
- Remove extra spaces and single characters
- Add `startseq` / `endseq` delimiters

---

## 📂 Project Structure

```
ImageCaptioning/
│
├── 📄 main.py                          # Streamlit app — upload image & generate caption
├── 📄 README.md                        # You are here!
│
├── 📁 models/
│   ├── 🧠 model.keras                  # Trained CNN-LSTM caption generation model (~52 MB)
│   ├── 🧠 feature_extractor.keras      # DenseNet201 feature extraction model (~76 MB)
│   ├── 📦 tokenizer.pkl                # Fitted tokenizer with vocabulary mappings
│   └── 📓 flickr8k-image-captioning-   # Full training notebook (Kaggle)
│          using-cnns-lstms.ipynb
│
└── 📁 input_image/                     # Directory for sample/test images
```

---

## 📈 Results

The model generates coherent, descriptive captions for a wide range of images. Some example outputs:

| Input | Generated Caption |
|---|---|
| 🏞️ Outdoor scene | *"a man is standing on a rock near the water"* |
| 🐕 Dog photo | *"a brown dog is running through the grass"* |
| 👶 People | *"a child in a red shirt is playing with a ball"* |

> **Note:** The model was trained on 8,000 images. Performance can be significantly improved by training on larger datasets like Flickr30k or MS-COCO.

---

## 🔮 Future Improvements

- [ ] **Attention Mechanism** — Implement Bahdanau or Transformer-based attention for better spatial focus
- [ ] **Larger Datasets** — Train on Flickr30k / MS-COCO for richer vocabulary and accuracy
- [ ] **BLEU Score Evaluation** — Add automated caption quality metrics
- [ ] **Beam Search Decoding** — Replace greedy decoding with beam search for better captions
- [ ] **Docker Support** — Containerize the app for one-command deployment
- [ ] **REST API** — Add a FastAPI endpoint for programmatic access

---

## 🧰 Tech Stack

<p align="center">
  <img src="https://img.shields.io/badge/Python-3776AB?style=flat-square&logo=python&logoColor=white" height="30"/>
  <img src="https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white" height="30"/>
  <img src="https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white" height="30"/>
  <img src="https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white" height="30"/>
  <img src="https://img.shields.io/badge/Matplotlib-11557C?style=flat-square&logo=matplotlib&logoColor=white" height="30"/>
  <img src="https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white" height="30"/>
</p>

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to open an issue or submit a pull request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## 📄 License

This project is licensed under the **MIT License** — see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Praveen Kumar**

- GitHub: [@prvn-kumar01](https://github.com/prvn-kumar01)

---

<p align="center">
  <em>If you found this project useful, consider giving it a ⭐ on GitHub!</em>
</p>
