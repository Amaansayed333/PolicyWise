import sqlite3
import hashlib
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

# Hash function to identify unique PDFs
def get_pdf_hash(file):
    file.seek(0)
    data = file.read()
    file.seek(0)  # reset pointer after reading
    return hashlib.md5(data).hexdigest()

# Database setup
conn = sqlite3.connect("summary_storage.db")
cursor = conn.cursor()

cursor.execute('''
    CREATE TABLE IF NOT EXISTS summaries (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        summary TEXT,
        hash TEXT UNIQUE
    )
''')
conn.commit()

# Streamlit app
st.title("📚 Smart PDF Chatbot")

uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
if uploaded_file:
    # Step 1: Hash the file
    pdf_hash = get_pdf_hash(uploaded_file)

    # Step 2: Check DB for previous summary
    cursor.execute('SELECT summary FROM summaries WHERE hash = ?', (pdf_hash,))
    stored_result = cursor.fetchone()

    # Step 3: Read and summarize
    pdf_text = read_pdf(uploaded_file)
    generated_summary = summarizer(pdf_text, max_length=300, min_length=50, do_sample=False)[0]['summary_text']

    # Step 4: Show newly generated summary
    st.subheader("🆕 Summary Generated This Time")
    st.success(generated_summary)

    # Step 5: Show previously stored summary (if any)
    if stored_result:
        st.subheader("📁 Summary Stored Earlier")
        st.info(stored_result[0])
    else:
        # Store this new summary
        try:
            cursor.execute('INSERT INTO summaries (summary, hash) VALUES (?, ?)', (generated_summary, pdf_hash))
            conn.commit()
        except sqlite3.IntegrityError:
            pass  # Already stored

    # Optional: Ask follow-up question
    question = st.text_input("Ask a follow-up question based on the PDF:")
    if question:
        result = qa_pipeline(question=question, context=pdf_text)
        st.info(f"**Answer:** {result['answer']}")
