import streamlit as st
import pdfplumber
import pytesseract
from PIL import Image
from transformers import pipeline

# Page setup
st.set_page_config(page_title="Document Summary Assistant", layout="centered")
st.title("📄 Document Summary Assistant")
st.write("Upload a **PDF or Image**, and get a smart summary automatically!")

# File uploader
uploaded_file = st.file_uploader("Upload your document (PDF or Image)", type=["pdf", "png", "jpg", "jpeg"])

text = ""

# Extract text from uploaded file
if uploaded_file is not None:
    file_type = uploaded_file.name.split(".")[-1].lower()

    if file_type == "pdf":
        with pdfplumber.open(uploaded_file) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ""

    elif file_type in ["png", "jpg", "jpeg"]:
        image = Image.open(uploaded_file)
        text = pytesseract.image_to_string(image)

    if not text.strip():
        st.warning("⚠️ No readable text found in the file. Try another document.")
    else:
        st.subheader("📘 Extracted Text Preview:")
        st.text_area("Extracted Text", text[:1500] + ("..." if len(text) > 1500 else ""), height=200, label_visibility="collapsed")

        st.subheader("🧩 Choose Summary Length:")
        choice = st.radio("Select summary length", ["Short", "Medium", "Long"], label_visibility="collapsed")

        if st.button("Generate Summary"):
            with st.spinner("⏳ Summarizing... Please wait"):
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

                max_length = {"Short": 60, "Medium": 120, "Long": 180}[choice]
                min_length = max_length // 3

                chunks = [text[i:i+1000] for i in range(0, len(text), 1000)]
                summary = ""

                for chunk in chunks:
                    summary_part = summarizer(chunk, max_length=max_length, min_length=min_length, do_sample=False)
                    summary += summary_part[0]['summary_text'] + " "

            st.success("✅ Summary Generated Successfully!")
            st.subheader("📝 Smart Summary:")
            st.write(summary.strip())

else:
    st.info("👆 Please upload a PDF or image to get started.")
