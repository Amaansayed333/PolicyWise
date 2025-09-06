import streamlit as st
import fitz  # PyMuPDF
from transformers import pipeline
import pyttsx3
import os
import sqlite3
from sentence_transformers import SentenceTransformer, util
import re
from datetime import datetime
import json

st.set_page_config(page_title="Insurance Policy AI Agent", page_icon="🛡️", layout="wide")

st.markdown(
    """
    <style>
    /* Full-screen animated aurora gradient */
    .stApp {
        background: radial-gradient(ellipse at bottom, #1b2735 0%, #090a0f 100%);
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
        0% { transform: scale(1) translate(0px, 0px); }
        50% { transform: scale(1.2) translate(20px, -30px); }
        100% { transform: scale(1) translate(0px, 0px); }
    }

    .policy-card {
        background: rgba(255, 255, 255, 0.1);
        backdrop-filter: blur(10px);
        border-radius: 15px;
        padding: 20px;
        margin: 10px 0;
        border: 1px solid rgba(255, 255, 255, 0.2);
    }

    .risk-high { 
        background: rgba(255, 82, 82, 0.2); 
        border-left: 4px solid #ff5252; 
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .risk-medium { 
        background: rgba(255, 193, 7, 0.2); 
        border-left: 4px solid #ffc107; 
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }
    
    .risk-low { 
        background: rgba(76, 175, 80, 0.2); 
        border-left: 4px solid #4caf50; 
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
    }

    .recommendation-positive {
        background: rgba(76, 175, 80, 0.3);
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #4caf50;
    }

    .recommendation-negative {
        background: rgba(255, 82, 82, 0.3);
        border-radius: 10px;
        padding: 15px;
        border: 2px solid #ff5252;
    }
    </style>
    <div class="aurora"></div>
    """,
    unsafe_allow_html=True
)

# Load NLP pipelines
@st.cache_resource
def load_models():
    qa_pipeline = pipeline("question-answering", model="distilbert-base-cased-distilled-squad")
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    classifier = pipeline("text-classification", model="nlptown/bert-base-multilingual-uncased-sentiment")
    embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
    return qa_pipeline, summarizer, classifier, embedding_model

qa_pipeline, summarizer, classifier, embedding_model = load_models()

# === PDF Reading ===
def read_pdf(file):
    doc = fitz.open(stream=file.read(), filetype="pdf")
    text = ""
    for page in doc:
        text += page.get_text()
    return text

# === Extract Important Dates ===
def extract_important_dates(text):
    # Common date patterns in insurance policies
    date_patterns = [
        r'(?:policy\s+)?(?:effective|start|commencement)\s+date[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(?:expiry|expiration|end|maturity)\s+date[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(?:renewal)\s+date[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(?:claim\s+)?(?:deadline|due)\s+date[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
        r'(?:premium\s+)?(?:payment|due)\s+date[:\s]*(\d{1,2}[\/\-\.]\d{1,2}[\/\-\.]\d{2,4})',
    ]
    
    important_dates = {}
    text_lower = text.lower()
    
    for pattern in date_patterns:
        matches = re.findall(pattern, text_lower, re.IGNORECASE)
        for match in matches:
            if 'effective' in pattern or 'start' in pattern or 'commencement' in pattern:
                important_dates['Policy Start Date'] = match
            elif 'expiry' in pattern or 'expiration' in pattern or 'end' in pattern or 'maturity' in pattern:
                important_dates['Policy End Date'] = match
            elif 'renewal' in pattern:
                important_dates['Renewal Date'] = match
            elif 'claim' in pattern or 'deadline' in pattern:
                important_dates['Claim Deadline'] = match
            elif 'premium' in pattern or 'payment' in pattern:
                important_dates['Premium Due Date'] = match
    
    return important_dates

# === Identify Risk Points ===
def identify_risk_points(text):
    risk_keywords = {
        'high_risk': [
            'exclusions', 'not covered', 'pre-existing conditions', 'waiting period', 
            'deductible', 'co-payment', 'limitations', 'restrictions', 'penalties',
            'cancellation', 'non-refundable', 'age limit', 'geographic restrictions'
        ],
        'medium_risk': [
            'subject to approval', 'medical examination required', 'documentation required',
            'proof of income', 'annual limit', 'sub-limits', 'depreciation', 
            'claim settlement ratio', 'network hospitals only'
        ],
        'low_risk': [
            'guaranteed renewal', 'no medical examination', 'cashless treatment',
            'worldwide coverage', 'lifetime renewability', 'no age limit',
            'immediate coverage', 'restoration benefit'
        ]
    }
    
    risk_points = {'high': [], 'medium': [], 'low': []}
    text_lower = text.lower()
    
    for risk_level, keywords in risk_keywords.items():
        for keyword in keywords:
            if keyword in text_lower:
                # Extract surrounding context
                start = max(0, text_lower.find(keyword) - 100)
                end = min(len(text), text_lower.find(keyword) + len(keyword) + 100)
                context = text[start:end].strip()
                
                if risk_level == 'high_risk':
                    risk_points['high'].append(f"⚠️ {keyword.title()}: {context[:150]}...")
                elif risk_level == 'medium_risk':
                    risk_points['medium'].append(f"⚡ {keyword.title()}: {context[:150]}...")
                else:
                    risk_points['low'].append(f"✅ {keyword.title()}: {context[:150]}...")
    
    return risk_points

# === Generate Policy Recommendation ===
def generate_recommendation(text, risk_points):
    # Analyze overall sentiment and risk factors
    high_risks = len(risk_points['high'])
    medium_risks = len(risk_points['medium'])
    low_risks = len(risk_points['low'])
    
    total_risks = high_risks + medium_risks
    total_benefits = low_risks
    
    # Calculate recommendation score
    risk_score = (high_risks * 3) + (medium_risks * 2) - (low_risks * 1)
    
    # Extract coverage amount and premium info if available
    coverage_patterns = [
        r'sum\s+(?:insured|assured)[:\s]*(?:rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)',
        r'coverage\s+amount[:\s]*(?:rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)',
        r'premium[:\s]*(?:rs\.?\s*)?(\d+(?:,\d+)*(?:\.\d+)?)'
    ]
    
    financial_info = []
    for pattern in coverage_patterns:
        matches = re.findall(pattern, text.lower())
        financial_info.extend(matches)
    
    # Generate recommendation
    if risk_score <= 0 and total_benefits >= 3:
        recommendation = "RECOMMENDED ✅"
        reason = f"This policy shows strong benefits ({total_benefits} positive factors) with minimal risks ({total_risks} risk factors). Good value proposition."
        recommendation_type = "positive"
    elif risk_score <= 3 and total_benefits >= 2:
        recommendation = "CONDITIONALLY RECOMMENDED ⚖️"
        reason = f"This policy has moderate risk-benefit balance. Consider the {high_risks} high-risk and {medium_risks} medium-risk factors before deciding."
        recommendation_type = "neutral"
    else:
        recommendation = "NOT RECOMMENDED ❌"
        reason = f"This policy has significant risks ({high_risks} high-risk factors) that outweigh the benefits. Consider alternatives."
        recommendation_type = "negative"
    
    return {
        'recommendation': recommendation,
        'reason': reason,
        'type': recommendation_type,
        'risk_score': risk_score,
        'financial_info': financial_info
    }

# === Text-to-Speech ===
def speak_text(text):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)  # Adjust speech rate
        engine.setProperty('volume', 0.9)
        engine.say(text)
        engine.runAndWait()
    except Exception as e:
        st.error(f"Speech synthesis error: {str(e)}")

def save_audio(text, filename="policy_analysis.mp3"):
    try:
        engine = pyttsx3.init()
        engine.setProperty('rate', 150)
        engine.save_to_file(text, filename)
        engine.runAndWait()
        return True
    except Exception as e:
        st.error(f"Audio saving error: {str(e)}")
        return False

# === SQLite Database Setup ===
def init_database():
    conn = sqlite3.connect("insurance_policies.db")
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS policy_analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            policy_name TEXT,
            summary TEXT,
            important_dates TEXT,
            risk_points TEXT,
            recommendation TEXT,
            content TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    return conn, cursor

conn, cursor = init_database()

# === Main Streamlit UI ===
st.title("🛡️ Insurance Policy AI Agent")
st.markdown("### *Intelligent Policy Analysis with Voice Assistance*")

# Sidebar for previous analyses
with st.sidebar:
    st.header("📋 Previous Analyses")
    cursor.execute('SELECT id, policy_name, created_at FROM policy_analyses ORDER BY created_at DESC LIMIT 10')
    previous_analyses = cursor.fetchall()
    
    if previous_analyses:
        for analysis_id, policy_name, created_at in previous_analyses:
            if st.button(f"📄 {policy_name[:20]}...", key=f"prev_{analysis_id}"):
                # Load previous analysis
                cursor.execute('SELECT * FROM policy_analyses WHERE id = ?', (analysis_id,))
                prev_data = cursor.fetchone()
                if prev_data:
                    st.session_state['show_previous'] = prev_data

# Main content area
uploaded_file = st.file_uploader("📤 Upload Insurance Policy PDF", type="pdf")

if uploaded_file:
    with st.spinner("🔍 Analyzing insurance policy..."):
        # Extract text from PDF
        policy_text = read_pdf(uploaded_file)
        policy_name = uploaded_file.name.replace('.pdf', '')
        
        # Generate comprehensive analysis
        with st.container():
            col1, col2 = st.columns([2, 1])
            
            with col1:
                # Policy Summary
                st.subheader("📋 Policy Summary")
                try:
                    # Split text into chunks for better summarization
                    chunks = [policy_text[i:i+1000] for i in range(0, len(policy_text), 800)]
                    summaries = []
                    
                    for chunk in chunks[:3]:  # Limit to first 3 chunks to avoid timeout
                        if len(chunk.strip()) > 50:
                            summary = summarizer(chunk, max_length=150, min_length=30, do_sample=False)[0]['summary_text']
                            summaries.append(summary)
                    
                    final_summary = " ".join(summaries)
                    st.markdown(f'<div class="policy-card">{final_summary}</div>', unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Summary generation failed: {str(e)}")
                    final_summary = "Unable to generate automatic summary. Please review the policy manually."
            
            with col2:
                # Important Dates
                st.subheader("📅 Important Dates")
                important_dates = extract_important_dates(policy_text)
                
                if important_dates:
                    for date_type, date_value in important_dates.items():
                        st.info(f"**{date_type}:** {date_value}")
                else:
                    st.warning("No specific dates found in the policy.")

        # Risk Analysis
        st.subheader("⚠️ Risk Analysis")
        risk_points = identify_risk_points(policy_text)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("#### 🔴 High Risk Factors")
            if risk_points['high']:
                for risk in risk_points['high'][:5]:  # Show top 5
                    st.markdown(f'<div class="risk-high">{risk}</div>', unsafe_allow_html=True)
            else:
                st.success("No high-risk factors identified!")
        
        with col2:
            st.markdown("#### 🟡 Medium Risk Factors")
            if risk_points['medium']:
                for risk in risk_points['medium'][:5]:
                    st.markdown(f'<div class="risk-medium">{risk}</div>', unsafe_allow_html=True)
            else:
                st.info("No medium-risk factors identified!")
        
        with col3:
            st.markdown("#### 🟢 Positive Factors")
            if risk_points['low']:
                for benefit in risk_points['low'][:5]:
                    st.markdown(f'<div class="risk-low">{benefit}</div>', unsafe_allow_html=True)
            else:
                st.warning("Limited positive factors identified.")

        # Recommendation
        st.subheader("🎯 AI Recommendation")
        recommendation = generate_recommendation(policy_text, risk_points)
        
        if recommendation['type'] == 'positive':
            st.markdown(f'<div class="recommendation-positive"><h3>{recommendation["recommendation"]}</h3><p>{recommendation["reason"]}</p></div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="recommendation-negative"><h3>{recommendation["recommendation"]}</h3><p>{recommendation["reason"]}</p></div>', unsafe_allow_html=True)

        # Check for similar policies
        current_embedding = embedding_model.encode(policy_text, convert_to_tensor=True)
        cursor.execute('SELECT content, policy_name, summary, recommendation FROM policy_analyses')
        stored_policies = cursor.fetchall()

        similar_policies = []
        for content, name, summary, rec in stored_policies:
            if content:
                stored_embedding = embedding_model.encode(content, convert_to_tensor=True)
                similarity = util.pytorch_cos_sim(current_embedding, stored_embedding).item()
                if similarity >= 0.75:  # 75% similarity threshold
                    similar_policies.append((name, summary, rec, similarity))

        # Show similar policies
        if similar_policies:
            st.subheader("📊 Similar Policies Previously Analyzed")
            for name, summary, rec, similarity in similar_policies[:3]:
                with st.expander(f"📄 {name} (Similarity: {similarity:.1%})"):
                    st.write(f"**Summary:** {summary}")
                    st.write(f"**Previous Recommendation:** {rec}")

        # Audio Features
        st.subheader("🔊 Audio Summary")
        audio_text = f"""
        Policy Analysis Summary:
        
        {final_summary}
        
        Key Risk Factors: {len(risk_points['high'])} high risk, {len(risk_points['medium'])} medium risk factors identified.
        
        Recommendation: {recommendation['recommendation']}
        {recommendation['reason']}
        """
        
        col1, col2 = st.columns(2)
        with col1:
            if st.button("🎤 Listen to Analysis"):
                speak_text(audio_text)
        
        with col2:
            if st.button("💾 Download Audio"):
                if save_audio(audio_text, f"{policy_name}_analysis.mp3"):
                    st.success("Audio saved successfully!")
                    if os.path.exists(f"{policy_name}_analysis.mp3"):
                        with open(f"{policy_name}_analysis.mp3", "rb") as audio_file:
                            st.download_button(
                                label="📥 Download Audio File",
                                data=audio_file.read(),
                                file_name=f"{policy_name}_analysis.mp3",
                                mime="audio/mp3"
                            )

        # Save to database
        try:
            cursor.execute('''
                INSERT INTO policy_analyses 
                (policy_name, summary, important_dates, risk_points, recommendation, content) 
                VALUES (?, ?, ?, ?, ?, ?)
            ''', (
                policy_name,
                final_summary,
                json.dumps(important_dates),
                json.dumps(risk_points),
                json.dumps(recommendation),
                policy_text
            ))
            conn.commit()
            st.success("✅ Analysis saved for future reference!")
        except Exception as e:
            st.error(f"Database error: {str(e)}")

        # Interactive Q&A
        st.subheader("❓ Ask Questions About This Policy")
        question = st.text_input("Enter your question about the insurance policy:")
        
        if question and question.lower() != 'exit':
            try:
                with st.spinner("🤔 Analyzing your question..."):
                    result = qa_pipeline(question=question, context=policy_text[:2000])  # Limit context
                    st.info(f"**Answer:** {result['answer']}")
                    
                    if st.button("🔊 Listen to Answer"):
                        speak_text(result['answer'])
            except Exception as e:
                st.error(f"Question answering error: {str(e)}")

# Show previous analysis if requested
if 'show_previous' in st.session_state:
    prev_data = st.session_state['show_previous']
    st.subheader(f"📋 Previous Analysis: {prev_data[1]}")
    st.write(f"**Summary:** {prev_data[2]}")
    
    if prev_data[3]:  # important_dates
        dates = json.loads(prev_data[3])
        st.write("**Important Dates:**", dates)
    
    if prev_data[5]:  # recommendation
        rec = json.loads(prev_data[5])
        st.write(f"**Recommendation:** {rec['recommendation']}")
        st.write(f"**Reason:** {rec['reason']}")
    
    del st.session_state['show_previous']