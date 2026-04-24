import streamlit as st
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
import re
import spacy
import fitz
import tempfile
import os

# ── Page Config ────────────────────────────────────────────────
st.set_page_config(
    page_title="ResumeMatch AI",
    page_icon="⚡",
    layout="centered"
)

# ── Custom CSS ─────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:wght@300;400;500&display=swap');

/* Base */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background-color: #0a0a0f;
    color: #e8e8f0;
}

/* Hide Streamlit branding */
#MainMenu, footer, header { visibility: hidden; }

/* Main container */
.block-container {
    max-width: 820px;
    padding: 3rem 2rem;
}

/* Hero header */
.hero {
    text-align: center;
    padding: 3rem 0 2rem 0;
    border-bottom: 1px solid #1e1e2e;
    margin-bottom: 2.5rem;
}

.hero-badge {
    display: inline-block;
    background: linear-gradient(135deg, #6366f1, #8b5cf6);
    color: white;
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    padding: 0.35rem 1rem;
    border-radius: 100px;
    margin-bottom: 1.2rem;
}

.hero h1 {
    font-family: 'Syne', sans-serif;
    font-size: 3rem;
    font-weight: 800;
    background: linear-gradient(135deg, #ffffff 0%, #a5b4fc 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
    margin: 0 0 0.8rem 0;
}

.hero p {
    color: #6b6b8a;
    font-size: 1rem;
    font-weight: 300;
    margin: 0;
}

/* Section labels */
.section-label {
    font-family: 'Syne', sans-serif;
    font-size: 0.75rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #6366f1;
    margin-bottom: 0.6rem;
    display: block;
}

/* Upload box */
[data-testid="stFileUploader"] {
    background: #11111f;
    border: 1.5px dashed #2a2a4a;
    border-radius: 12px;
    padding: 1rem;
    transition: border-color 0.2s;
}

[data-testid="stFileUploader"]:hover {
    border-color: #6366f1;
}

/* Text area */
[data-testid="stTextArea"] textarea {
    background: #11111f !important;
    border: 1.5px solid #2a2a4a !important;
    border-radius: 12px !important;
    color: #e8e8f0 !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.9rem !important;
    resize: none !important;
    transition: border-color 0.2s !important;
}

[data-testid="stTextArea"] textarea:focus {
    border-color: #6366f1 !important;
    box-shadow: 0 0 0 3px rgba(99,102,241,0.15) !important;
}

/* Analyze button */
[data-testid="stButton"] > button {
    width: 100%;
    background: linear-gradient(135deg, #6366f1, #8b5cf6) !important;
    color: white !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.05em !important;
    border: none !important;
    border-radius: 12px !important;
    padding: 0.85rem 2rem !important;
    cursor: pointer !important;
    transition: opacity 0.2s, transform 0.1s !important;
    margin-top: 0.5rem !important;
}

[data-testid="stButton"] > button:hover {
    opacity: 0.9 !important;
    transform: translateY(-1px) !important;
}

/* Score card */
.score-card {
    background: linear-gradient(135deg, #11111f 0%, #16162a 100%);
    border: 1px solid #2a2a4a;
    border-radius: 20px;
    padding: 2.5rem;
    text-align: center;
    margin: 2rem 0;
    position: relative;
    overflow: hidden;
}

.score-card::before {
    content: '';
    position: absolute;
    top: -60px;
    left: 50%;
    transform: translateX(-50%);
    width: 200px;
    height: 200px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    pointer-events: none;
}

.score-number {
    font-family: 'Syne', sans-serif;
    font-size: 4.5rem;
    font-weight: 800;
    line-height: 1;
    margin-bottom: 0.5rem;
}

.score-label {
    font-size: 0.85rem;
    color: #6b6b8a;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    font-weight: 500;
}

.score-verdict {
    display: inline-block;
    margin-top: 1.2rem;
    padding: 0.4rem 1.2rem;
    border-radius: 100px;
    font-size: 0.8rem;
    font-weight: 600;
    letter-spacing: 0.05em;
}

.verdict-strong {
    background: rgba(34, 197, 94, 0.15);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.3);
}

.verdict-moderate {
    background: rgba(234, 179, 8, 0.15);
    color: #eab308;
    border: 1px solid rgba(234, 179, 8, 0.3);
}

.verdict-weak {
    background: rgba(239, 68, 68, 0.15);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
}

/* Progress bar */
.progress-bar-bg {
    background: #1e1e2e;
    border-radius: 100px;
    height: 8px;
    margin: 1.5rem 0 0 0;
    overflow: hidden;
}

.progress-bar-fill {
    height: 100%;
    border-radius: 100px;
    transition: width 1s ease;
}

/* Keyword cards */
.keyword-section {
    background: #11111f;
    border: 1px solid #2a2a4a;
    border-radius: 16px;
    padding: 1.5rem;
    height: 100%;
}

.keyword-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 1rem;
    display: flex;
    align-items: center;
    gap: 0.5rem;
}

.keyword-tag {
    display: inline-block;
    padding: 0.3rem 0.75rem;
    border-radius: 6px;
    font-size: 0.8rem;
    font-weight: 500;
    margin: 0.2rem;
}

.tag-found {
    background: rgba(34, 197, 94, 0.1);
    color: #22c55e;
    border: 1px solid rgba(34, 197, 94, 0.2);
}

.tag-missing {
    background: rgba(239, 68, 68, 0.1);
    color: #ef4444;
    border: 1px solid rgba(239, 68, 68, 0.2);
}

/* Tip box */
.tip-box {
    background: rgba(99, 102, 241, 0.08);
    border: 1px solid rgba(99, 102, 241, 0.2);
    border-radius: 12px;
    padding: 1rem 1.25rem;
    font-size: 0.85rem;
    color: #a5b4fc;
    margin-top: 1.5rem;
    line-height: 1.6;
}

/* Divider */
.custom-divider {
    border: none;
    border-top: 1px solid #1e1e2e;
    margin: 2rem 0;
}

/* Spinner */
[data-testid="stSpinner"] {
    color: #6366f1 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Load Models ────────────────────────────────────────────────
@st.cache_resource
def load_models():
    model = SentenceTransformer('all-MiniLM-L6-v2')
    kw_model = KeyBERT(model=model)
    nlp_spacy = spacy.load('en_core_web_sm')
    return model, kw_model, nlp_spacy

model, kw_model, nlp_spacy = load_models()

# ── Helper Functions ───────────────────────────────────────────
def read_pdf(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        tmp_path = tmp.name
    pdf = fitz.open(tmp_path)
    text = ""
    for page in pdf:
        text += page.get_text()
    pdf.close()
    os.unlink(tmp_path)
    return text.strip()

def is_skill_word(word):
    doc = nlp_spacy(word)
    for token in doc:
        if token.pos_ in ('NOUN', 'PROPN'):
            return True
    return False

def extract_keywords(text, top_n=20):
    keybert_results = kw_model.extract_keywords(
        text,
        keyphrase_ngram_range=(1, 1),
        stop_words='english',
        top_n=50,
        use_mmr=True,
        diversity=0.5
    )
    keybert_words = [
        word.lower() for word, score in keybert_results
        if score > 0.30
        and len(word) > 2
        and is_skill_word(word)
    ]
    abbreviations = list(set([
        a.lower() for a in re.findall(r'\b[A-Z]{2,}\b', text)
    ]))
    seen = set()
    all_keywords = []
    for kw in abbreviations + keybert_words:
        if kw not in seen:
            seen.add(kw)
            all_keywords.append(kw)
    return all_keywords[:top_n]

def find_keyword_gaps(job_keywords, resume_text):
    resume_lower = resume_text.lower()
    found, missing = [], []
    for skill in job_keywords:
        if skill.lower() in resume_lower:
            found.append(skill)
        else:
            missing.append(skill)
    return found, missing

def get_score_color(score):
    if score >= 75:
        return "#22c55e"
    elif score >= 50:
        return "#eab308"
    else:
        return "#ef4444"

# ── UI ─────────────────────────────────────────────────────────

# Hero
st.markdown("""
<div class="hero">
    <div class="hero-badge">⚡ AI Powered</div>
    <h1>ResumeMatch AI</h1>
    <p>Upload your resume and paste a job description — get an instant match score and skill gap report.</p>
</div>
""", unsafe_allow_html=True)

# Inputs
col1, col2 = st.columns(2)

with col1:
    st.markdown('<span class="section-label">📎 Your Resume</span>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("", type=["pdf"], label_visibility="collapsed")

with col2:
    st.markdown('<span class="section-label">💼 Job Description</span>', unsafe_allow_html=True)
    job_description = st.text_area(
        "",
        height=160,
        placeholder="Paste the full job description here...",
        label_visibility="collapsed"
    )

# Analyze button
analyze = st.button("⚡ Analyze Match", use_container_width=True)

# Results
if analyze:
    if not uploaded_file:
        st.error("Please upload your resume PDF.")
    elif not job_description.strip():
        st.error("Please paste a job description.")
    else:
        with st.spinner("Analyzing your resume..."):
            resume_text = read_pdf(uploaded_file)
            resume_embedding = model.encode([resume_text])
            job_embedding = model.encode([job_description])
            score = cosine_similarity(resume_embedding, job_embedding)[0][0]
            match_percentage = round(float(score) * 100, 2)
            job_keywords = extract_keywords(job_description, top_n=20)
            found, missing = find_keyword_gaps(job_keywords, resume_text)

        # Score card
        color = get_score_color(match_percentage)
        if match_percentage >= 75:
            verdict_class = "verdict-strong"
            verdict_text = "Strong Match"
            verdict_emoji = "🟢"
        elif match_percentage >= 50:
            verdict_class = "verdict-moderate"
            verdict_text = "Moderate Match"
            verdict_emoji = "🟡"
        else:
            verdict_class = "verdict-weak"
            verdict_text = "Weak Match"
            verdict_emoji = "🔴"

        st.markdown(f"""
        <div class="score-card">
            <div class="score-number" style="color: {color}">{match_percentage:.1f}%</div>
            <div class="score-label">Overall Match Score</div>
            <div class="score-verdict {verdict_class}">{verdict_emoji} {verdict_text}</div>
            <div class="progress-bar-bg">
                <div class="progress-bar-fill" style="width: {match_percentage}%; background: linear-gradient(90deg, {color}88, {color});"></div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Keywords
        st.markdown('<hr class="custom-divider">', unsafe_allow_html=True)

        k_col1, k_col2 = st.columns(2)

        with k_col1:
            found_tags = "".join([
                f'<span class="keyword-tag tag-found">{skill}</span>'
                for skill in found
            ]) if found else '<span style="color:#6b6b8a; font-size:0.85rem">None found</span>'

            st.markdown(f"""
            <div class="keyword-section">
                <div class="keyword-section-title" style="color: #22c55e">
                    ✅ Found in Resume ({len(found)})
                </div>
                {found_tags}
            </div>
            """, unsafe_allow_html=True)

        with k_col2:
            missing_tags = "".join([
                f'<span class="keyword-tag tag-missing">{skill}</span>'
                for skill in missing
            ]) if missing else '<span style="color:#22c55e; font-size:0.85rem">🎉 No gaps found!</span>'

            st.markdown(f"""
            <div class="keyword-section">
                <div class="keyword-section-title" style="color: #ef4444">
                    ❌ Missing Keywords ({len(missing)})
                </div>
                {missing_tags}
            </div>
            """, unsafe_allow_html=True)

        # Tip
        st.markdown("""
        <div class="tip-box">
            💡 <strong>Pro tip:</strong> Add the missing keywords naturally into your resume
            to improve your ATS score. Don't just list them — incorporate them into your
            experience descriptions and project summaries.
        </div>
        """, unsafe_allow_html=True)