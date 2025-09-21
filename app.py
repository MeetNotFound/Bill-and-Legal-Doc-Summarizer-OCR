import streamlit as st
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from PIL import Image
import pytesseract
import tempfile
import os

# ==============================
# App Config
# ==============================
st.set_page_config(page_title="Summify AI", layout="wide")
st.title("Summify AI - Document & Image Summarizer")

# ==============================
# Load Model + Tokenizer from Hugging Face
# ==============================
@st.cache_resource
def load_model():
    MODEL_HF_REPO = "MeetNotFound/Bill-and-Legal-Doc-Summarizer"
    HF_TOKEN = os.environ.get("HF_TOKEN")  # must set this in Streamlit Secrets

    tokenizer = AutoTokenizer.from_pretrained(MODEL_HF_REPO, use_auth_token=HF_TOKEN)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_HF_REPO, use_auth_token=HF_TOKEN)
    return tokenizer, model

with st.spinner("Loading model from Hugging Face..."):
    tokenizer, model = load_model()
st.success("Model loaded successfully!")

# ==============================
# Helper Functions
# ==============================
def extract_text_from_pdf(uploaded_pdf):
    """Extract text from PDF using PyPDF2"""
    temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
    temp_file.write(uploaded_pdf.read())
    temp_file.flush()

    from PyPDF2 import PdfReader
    pdf_reader = PdfReader(temp_file.name)
    text = ""
    for page in pdf_reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"

    temp_file.close()
    os.remove(temp_file.name)
    return text.strip()

def extract_text_from_image(uploaded_image):
    """Extract text from image using pytesseract OCR"""
    image = Image.open(uploaded_image)
    text = pytesseract.image_to_string(image)
    return text.strip()

def summarize_text(text):
    """Summarize extracted text using your Hugging Face model"""
    inputs = tokenizer(text, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(
        **inputs,
        max_length=128,
        min_length=30,
        length_penalty=2.0,
        num_beams=4,
        early_stopping=True
    )
    return tokenizer.decode(summary_ids[0], skip_special_tokens=True)

# ==============================
# File Uploader
# ==============================
uploaded_file = st.file_uploader("Upload a PDF or Image", type=["pdf", "png", "jpg", "jpeg"])

if uploaded_file is not None:
    with st.spinner("Processing file..."):
        if uploaded_file.type == "application/pdf":
            st.write("**PDF detected — extracting text**")
            extracted_text = extract_text_from_pdf(uploaded_file)

        elif uploaded_file.type.startswith("image/"):
            st.write("**Image detected — running OCR**")
            extracted_text = extract_text_from_image(uploaded_file)

        else:
            st.error("Unsupported file type!")
            st.stop()

        if not extracted_text:
            st.warning("No text found in the file.")
        else:
            st.text_area("Extracted Text", extracted_text, height=300)

            if st.button("🔍 Summarize"):
                with st.spinner("Generating summary..."):
                    summary = summarize_text(extracted_text)
                st.success("Summary Generated!")
                st.text_area("Summary", summary, height=200)