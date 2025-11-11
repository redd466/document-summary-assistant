import streamlit as st
import pdfplumber
from PIL import Image
import pytesseract
from transformers import pipeline

# Page configuration
st.set_page_config(page_title="Document Summary Assistant", layout="centered")
st.title("📄 Document Summary Assistant")
st.write("Upload a **PDF** or **Image** file to extract text and generate a smart summary.")

# File uploader
uploaded_file = st.file_uploader("Choose a file", type=["pdf", "png", "jpg", "jpeg"])

# Function to extract text from PDF
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

# Function to extract text from Image
def extract_text_from_image(file):
    image = Image.open(file)
    text = pytesseract.image_to_string(image)
    return text

# Summarization model (smaller for faster load)
@st.cache_resource
def load_summarizer():
    return pipeline("summarization", model="sshleifer/distilbart-cnn-6-6")

# Main logic
if uploaded_file is not None:
    if uploaded_file.name.endswith(".pdf"):
        text = extract_text_from_pdf(uploaded_file)
    else:
        text = extract_text_from_image(uploaded_file)

    if text.strip() == "":
        st.warning("No text could be extracted from this file.")
    else:
        st.subheader("Extracted Text")
        st.text_area(
            "Extracted Text",
            text[:1500] + ("..." if len(text) > 1500 else ""),
            height=200,
            label_visibility="collapsed"
        )

        choice = st.radio(
            "Select summary length",
            ["Short", "Medium", "Long"],
            label_visibility="collapsed"
        )

        if st.button("Generate Summary"):
            summarizer = load_summarizer()

            if choice == "Short":
                max_len, min_len = 60, 20
            elif choice == "Medium":
                max_len, min_len = 120, 40
            else:
                max_len, min_len = 200, 80

            with st.spinner("Summarizing... please wait ⏳"):
                summary = summarizer(
                    text, max_length=max_len, min_length=min_len, do_sample=False
                )[0]["summary_text"]

            st.success("Summary generated successfully!")
            st.subheader("📘 Summary:")
            st.write(summary)
else:
    st.info("Please upload a file to begin.")
