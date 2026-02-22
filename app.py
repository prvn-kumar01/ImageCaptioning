import streamlit as st
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from PIL import Image
import pickle
import os
import tempfile

os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

# ──────────────────────────────────────────────────────────────
# Page Config
# ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# ──────────────────────────────────────────────────────────────
# Custom CSS — Premium dark theme with glassmorphism
# ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ─── Root variables ─── */
:root {
    --bg-primary: #0f0f1a;
    --bg-card: rgba(255, 255, 255, 0.04);
    --bg-card-hover: rgba(255, 255, 255, 0.07);
    --accent-gradient: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    --accent-blue: #667eea;
    --accent-purple: #764ba2;
    --text-primary: #f0f0f5;
    --text-secondary: #a0a0b8;
    --border-color: rgba(255, 255, 255, 0.08);
    --glow: 0 0 40px rgba(102, 126, 234, 0.15);
}

/* ─── Global ─── */
.stApp {
    background: var(--bg-primary) !important;
    font-family: 'Inter', sans-serif !important;
}

.stApp > header { background: transparent !important; }

/* ─── Hide Streamlit defaults ─── */
#MainMenu, footer, .stDeployButton { display: none !important; }

/* ─── Hero section ─── */
.hero {
    text-align: center;
    padding: 2.5rem 1rem 1.5rem;
}
.hero-icon {
    font-size: 3.8rem;
    margin-bottom: 0.6rem;
    animation: float 3s ease-in-out infinite;
}
@keyframes float {
    0%, 100% { transform: translateY(0); }
    50% { transform: translateY(-10px); }
}
.hero h1 {
    font-size: 2.6rem;
    font-weight: 800;
    background: var(--accent-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    letter-spacing: -0.02em;
}
.hero p {
    color: var(--text-secondary);
    font-size: 1.05rem;
    margin-top: 0.5rem;
    font-weight: 300;
}

/* ─── Glass card ─── */
.glass-card {
    background: var(--bg-card);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border-color);
    border-radius: 20px;
    padding: 2rem;
    margin: 1rem 0;
    box-shadow: var(--glow);
    transition: all 0.3s ease;
}
.glass-card:hover {
    background: var(--bg-card-hover);
    box-shadow: 0 0 60px rgba(102, 126, 234, 0.2);
}

/* ─── Upload area ─── */
.upload-zone {
    border: 2px dashed rgba(102, 126, 234, 0.35);
    border-radius: 16px;
    padding: 2.5rem 1.5rem;
    text-align: center;
    transition: all 0.3s ease;
    background: rgba(102, 126, 234, 0.03);
}
.upload-zone:hover {
    border-color: var(--accent-blue);
    background: rgba(102, 126, 234, 0.06);
}
.upload-icon { font-size: 2.8rem; margin-bottom: 0.5rem; }
.upload-text {
    color: var(--text-secondary);
    font-size: 0.95rem;
}
.upload-text strong { color: var(--accent-blue); }

/* ─── Caption result ─── */
.caption-result {
    background: linear-gradient(135deg, rgba(102,126,234,0.12) 0%, rgba(118,75,162,0.12) 100%);
    border: 1px solid rgba(102, 126, 234, 0.25);
    border-radius: 16px;
    padding: 1.5rem 2rem;
    margin-top: 1.5rem;
    text-align: center;
    animation: fadeSlideUp 0.6s ease forwards;
}
.caption-label {
    font-size: 0.75rem;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--accent-blue);
    font-weight: 600;
    margin-bottom: 0.5rem;
}
.caption-text {
    font-size: 1.35rem;
    font-weight: 500;
    color: var(--text-primary);
    font-style: italic;
    line-height: 1.5;
}

@keyframes fadeSlideUp {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* ─── Image display ─── */
.image-frame {
    border-radius: 16px;
    overflow: hidden;
    border: 1px solid var(--border-color);
    box-shadow: 0 8px 32px rgba(0,0,0,0.3);
    animation: fadeSlideUp 0.4s ease forwards;
}
.image-frame img {
    width: 100%;
    display: block;
}

/* ─── Pipeline badges ─── */
.pipeline {
    display: flex;
    gap: 0.5rem;
    justify-content: center;
    flex-wrap: wrap;
    margin: 1.2rem 0;
}
.badge {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    padding: 0.4rem 0.9rem;
    border-radius: 99px;
    font-size: 0.78rem;
    font-weight: 500;
    border: 1px solid var(--border-color);
    color: var(--text-secondary);
    background: var(--bg-card);
}
.badge-arrow { color: var(--accent-blue); }

/* ─── Info row ─── */
.info-row {
    display: flex;
    gap: 1rem;
    margin-top: 1.2rem;
    flex-wrap: wrap;
}
.info-item {
    flex: 1;
    min-width: 140px;
    background: var(--bg-card);
    border: 1px solid var(--border-color);
    border-radius: 12px;
    padding: 1rem;
    text-align: center;
}
.info-item .label {
    font-size: 0.7rem;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}
.info-item .value {
    font-size: 1rem;
    font-weight: 600;
    color: var(--text-primary);
}

/* ─── Footer ─── */
.app-footer {
    text-align: center;
    padding: 2rem 0 1rem;
    color: var(--text-secondary);
    font-size: 0.8rem;
}
.app-footer a {
    color: var(--accent-blue);
    text-decoration: none;
    font-weight: 500;
}
.app-footer a:hover { text-decoration: underline; }

/* ─── Streamlit overrides ─── */
.stFileUploader > div { border: none !important; }
.stFileUploader label { display: none !important; }
.stSpinner > div { border-top-color: var(--accent-blue) !important; }
div[data-testid="stImage"] { border-radius: 16px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────
# Model loading — cached for performance
# ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_models():
    """Load caption model, feature extractor and tokenizer once."""
    caption_model = load_model("models/model.keras")
    feature_extractor = load_model("models/feature_extractor.keras")
    with open("models/tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)
    return caption_model, feature_extractor, tokenizer


def generate_caption(image_path, caption_model, feature_extractor, tokenizer, max_length=34, img_size=224):
    """Extract features and generate a caption word-by-word."""
    img = load_img(image_path, target_size=(img_size, img_size))
    img_arr = img_to_array(img) / 255.0
    img_arr = np.expand_dims(img_arr, axis=0)
    features = feature_extractor.predict(img_arr, verbose=0)

    in_text = "startseq"
    for _ in range(max_length):
        seq = tokenizer.texts_to_sequences([in_text])[0]
        seq = pad_sequences([seq], maxlen=max_length)
        yhat = caption_model.predict([features, seq], verbose=0)
        idx = np.argmax(yhat)
        word = tokenizer.index_word.get(idx)
        if word is None:
            break
        in_text += " " + word
        if word == "endseq":
            break

    return in_text.replace("startseq", "").replace("endseq", "").strip()


# ──────────────────────────────────────────────────────────────
# Hero header
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-icon">🖼️</div>
    <h1>Image Caption Generator</h1>
    <p>Upload an image and let AI describe it using a DenseNet201 + LSTM pipeline</p>
</div>
""", unsafe_allow_html=True)

# Pipeline visualization
st.markdown("""
<div class="pipeline">
    <span class="badge">📷 Upload</span>
    <span class="badge-arrow">→</span>
    <span class="badge">🧠 DenseNet201</span>
    <span class="badge-arrow">→</span>
    <span class="badge">🔗 LSTM Decoder</span>
    <span class="badge-arrow">→</span>
    <span class="badge">📝 Caption</span>
</div>
""", unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Upload section
# ──────────────────────────────────────────────────────────────
st.markdown('<div class="glass-card">', unsafe_allow_html=True)

st.markdown("""
<div class="upload-zone">
    <div class="upload-icon">📤</div>
    <p class="upload-text">
        <strong>Drag & drop</strong> an image below, or click to browse<br>
        Supports JPG, JPEG, PNG
    </p>
</div>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader(
    "Upload Image",
    type=["jpg", "jpeg", "png"],
    label_visibility="collapsed",
)

st.markdown('</div>', unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Processing & results
# ──────────────────────────────────────────────────────────────
if uploaded_file is not None:
    image = Image.open(uploaded_file)

    # Show uploaded image
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.image(image, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Save to temp file for model input
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp:
        image.save(tmp, format="JPEG")
        tmp_path = tmp.name

    # Generate caption
    with st.spinner("🧠  Analyzing image and generating caption…"):
        caption_model, feature_extractor, tokenizer = load_models()
        caption = generate_caption(tmp_path, caption_model, feature_extractor, tokenizer)

    # Clean up temp file
    os.unlink(tmp_path)

    # Display caption
    st.markdown(f"""
    <div class="caption-result">
        <div class="caption-label">✨ Generated Caption</div>
        <div class="caption-text">"{caption}"</div>
    </div>
    """, unsafe_allow_html=True)

    # Image info
    w, h = image.size
    size_kb = uploaded_file.size / 1024
    st.markdown(f"""
    <div class="info-row">
        <div class="info-item">
            <div class="label">Dimensions</div>
            <div class="value">{w} × {h}</div>
        </div>
        <div class="info-item">
            <div class="label">File Size</div>
            <div class="value">{size_kb:.1f} KB</div>
        </div>
        <div class="info-item">
            <div class="label">Model</div>
            <div class="value">DenseNet201</div>
        </div>
        <div class="info-item">
            <div class="label">Decoder</div>
            <div class="value">LSTM-256</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ──────────────────────────────────────────────────────────────
# Footer
# ──────────────────────────────────────────────────────────────
st.markdown("""
<div class="app-footer">
    Built with ❤️ using <a href="https://streamlit.io" target="_blank">Streamlit</a> &
    <a href="https://www.tensorflow.org" target="_blank">TensorFlow</a><br>
    <a href="https://github.com/prvn-kumar01/ImageCaptioning" target="_blank">GitHub ↗</a>
</div>
""", unsafe_allow_html=True)
