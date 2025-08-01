import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline

# Load QA pipeline
qa_pipeline = pipeline("question-answering")
summarizer = pipeline("summarization")

# Read PDF function
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Streamlit app
st.title("📚 Smart PDF Chatbot")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_text = read_pdf(uploaded_file)
    st.subheader("📄 PDF Summary")
    summary = summarizer(pdf_text, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
    st.success(summary)

    st.subheader("❓ Ask a Question")
    question = st.text_input("Type your question about the PDF")

    if question:
        result = qa_pipeline(question=question, context=pdf_text)
        st.info(f"**Answer:** {result['answer']}")
