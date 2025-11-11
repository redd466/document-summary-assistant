import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline

st.set_page_config(page_title="Document Summary Assistant", layout="centered")
st.title("📄 Document Summary Assistant")
st.write("Upload a **PDF** or **Image** file to generate a smart summary.")

# Upload section
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])

def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-12-6")

summarizer = load_summarizer()

def summarize_text(text, ratio=0.3):
    try:
        # calculate summary length dynamically
        max_len = int(len(text.split()) * ratio)
        max_len = max(60, min(max_len, 500))  # limit the length between 60 and 500 words
        result = summarizer(text, max_length=max_len, min_length=30, do_sample=False)
        return result[0]['summary_text']
    except Exception as e:
        return f"Could not generate summary: {e}"


if uploaded_file:
    file_type = uploaded_file.name.split(".")[-1].lower()

    with st.spinner("Extracting text..."):
        if file_type == "pdf":
            text = extract_text_from_pdf(uploaded_file)
        else:
            text = extract_text_from_image(uploaded_file)

    st.subheader("Extracted Text:")
    st.text_area("", text[:1500] + ("..." if len(text) > 1500 else ""), height=200)

    st.subheader("Choose Summary Length:")
    choice = st.radio("", ["Short", "Medium", "Long"])

    if st.button("Generate Summary"):
        with st.spinner("Summarizing..."):
            ratio = 0.2 if choice == "Short" else 0.3 if choice == "Medium" else 0.5
            summary = summarize_text(text, ratio)
        st.success("✅ Summary Generated Successfully!")
        st.text_area("📋 Summary:", summary, height=250)
