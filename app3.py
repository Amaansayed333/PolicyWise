import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import pyttsx3
import os
import sqlite3
from sentence_transformers import SentenceTransformer, util

st.markdown(
    """
    <style>
    /* Full-screen animated aurora gradient */
    body {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
        overflow: hidden;
    }

    .aurora {
        position: fixed;
        top: 0;
        left: 0;
        height: 100%;
        width: 100%;
        z-index: -1;
        background: radial-gradient(ellipse at top left, #6a3093, transparent),
                    radial-gradient(ellipse at bottom right, #00c9ff, transparent),
                    radial-gradient(ellipse at center, #b721ff33, transparent);
        animation: moveAurora 15s ease-in-out infinite;
        background-blend-mode: screen;
        opacity: 0.35;
        filter: blur(100px);
    }

    @keyframes moveAurora {
        0% {
            transform: scale(1) translate(0px, 0px);
        }
        50% {
            transform: scale(1.2) translate(20px, -30px);
        }
        100% {
            transform: scale(1) translate(0px, 0px);
        }
    }
    </style>
    <div class="aurora"></div>
    """,
    unsafe_allow_html=True
)


# Load NLP pipelines
qa_pipeline = pipeline("question-answering")
summarizer = pipeline("summarization")
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Fast + accurate

# === PDF Reading ===
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === Use pyttsx3 to speak text ===
def speak_text(text):
    engine = pyttsx3.init()
    engine.say(text)
    engine.runAndWait()

# === Save speech to audio file ===
def save_audio(text, filename="summary_audio.mp3"):
    engine = pyttsx3.init()
    engine.save_to_file(text, filename)
    engine.runAndWait()

# === SQLite Setup ===
conn = sqlite3.connect("summary_storage.db")
cursor = conn.cursor()
cursor.execute('''
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        summary TEXT,
        content TEXT
    )
''')
conn.commit()

# === Streamlit UI ===
st.title("📚 Smart PDF Chatbot with Voice & Memory (Semantic Matching)")

st.markdown(
    """<style>...your aurora CSS here...</style><div class="aurora"></div>""",
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    pdf_text = read_pdf(uploaded_file)

    # Generate new summary
    summary = summarizer(pdf_text, max_length=300, min_length=50, do_sample=False)[0]['summary_text']
    st.subheader("🆕 New Summary Generated")
    st.success(summary)

    # Embed current text
    current_embedding = embedding_model.encode(pdf_text, convert_to_tensor=True)

    # Check for similar content in DB
    cursor.execute('SELECT content, summary FROM summaries')
    stored_data = cursor.fetchall()

    most_similar = None
    highest_score = 0

    for content, old_summary in stored_data:
        stored_embedding = embedding_model.encode(content, convert_to_tensor=True)
        similarity = util.pytorch_cos_sim(current_embedding, stored_embedding).item()
        if similarity > highest_score:
            highest_score = similarity
            most_similar = old_summary

    # Show previous summary if similarity high
    if most_similar and highest_score >= 0.85:
        st.subheader(f"📁 Previously Stored Similar Summary (Similarity: {highest_score:.2f})")
        st.info(most_similar)
    else:
        # Save new summary to DB
        cursor.execute('INSERT INTO summaries (summary, content) VALUES (?, ?)', (summary, pdf_text))
        conn.commit()

    # Voice Output
    st.subheader("🔈 AI Speaking Summary...")
    speak_text(summary)
    audio_file = "summary_audio.mp3"
    save_audio(summary, audio_file)

    if os.path.exists(audio_file):
        st.audio(audio_file, format="audio/mp3")

    # Question Answering
    st.subheader("❓ Ask Questions About the PDF")
    question = st.text_input("Enter your question (or type 'exit' to stop):")

    if question.lower() == 'exit':
        st.warning("You typed 'exit'. Interaction stopped. 👋")
    elif question:
        result = qa_pipeline(question=question, context=pdf_text)
        st.info(f"**Answer:** {result['answer']}")
