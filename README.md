# DocMind
🧠 DocMind – The AI That Remembers What You Read

    DocMind is an intelligent document assistant that not only summarizes documents, but also remembers what it's read.
    It uses AI to generate concise summaries, stores them in a memory database, and recommends previously seen similar documents when a new one is uploaded — giving you instant context and insight.

✅ Key Features:

    📄 Upload any document and get a clean AI-generated summary

    🧠 Memory-powered engine — stores summaries in SQLite for future use

    🧭 When a new document is uploaded, detects similar summaries from past uploads

    ❓ Built-in question-answering system lets you interact with the content

    🔊 Offline Text-to-Speech reads the summary aloud

    💡 Ideal for researchers, students, analysts, or anyone managing lots of PDFs

🔧 Tech Stack
Feature	Tool
PDF Reading	PyMuPDF (fitz)
Summarization & QA	HuggingFace Transformers
Memory Storage	SQLite3
Frontend UI	Streamlit
TTS	pyttsx3 (offline)
