import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import pyttsx3
import os

# Load QA and summarization pipelines
qa_pipeline = pipeline("question-answering")
summarizer = pipeline("summarization")

# Function to read PDF
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# Function to speak text using pyttsx3
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# Function to save spoken text to audio file
def save_audio(text, filename="summary_audio.mp3"):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()

# Streamlit UI
st.title("📚 Smart PDF Chatbot with Voice Summary")

# Upload PDF
uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_text = read_pdf(uploaded_file)

    # Summarize PDF
    st.subheader("📄 PDF Summary")
    summary = summarizer(pdf_text, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
    st.success(summary)

    # Speak summary aloud
    st.subheader("🔈 AI Speaking Summary...")
    speak_text(summary)  # Speak via system TTS

    # Save and play audio file in Streamlit
    audio_file = "summary_audio.mp3"
    save_audio(summary, audio_file)

    if os.path.exists(audio_file):
        st.audio(audio_file, format="audio/mp3")

    # Question Answering
    st.subheader("❓ Ask a Question About the PDF")
    question = st.text_input("Enter your question (or type 'exit' to stop):")

    if question.lower() == 'exit':
        st.warning("You typed 'exit'. Interaction stopped. 👋")
    elif question:
        result = qa_pipeline(question=question, context=pdf_text)
        st.info(f"**Answer:** {result['answer']}")
