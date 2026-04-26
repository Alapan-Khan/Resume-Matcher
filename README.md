# ⚡ ResumeMatch AI
> Upload your resume + paste any job description → get an instant AI-powered match score and skill gap report.

🔗 **[Live Demo]- https://resume-matcher-d5vcchjefqttllwbo7ewyw.streamlit.app/



## The Problem It Solves
Most resumes get rejected by ATS systems before a human even reads them — because they're missing the right keywords. ResumeMatch AI tells you exactly what's missing and how well you match, before you apply.

## How It Works
1. **Upload** your resume as a PDF
2. **Paste** any job description
3. **Get** a match score (0–100%) + a list of found vs missing keywords

The match score uses **semantic similarity** (Sentence Transformers + cosine similarity) — so it understands meaning, not just exact word matches. Keyword extraction uses **KeyBERT + NLTK POS tagging** to surface only real skills, not generic filler words. Works for any job in any industry.

## Tech Stack
| NLP & Scoring | Sentence Transformers, KeyBERT, NLTK, scikit-learn |
| PDF Parsing | PyMuPDF |
| Frontend & Deployment | Streamlit, Streamlit Cloud |

## Run Locally
bash--
git clone https://github.com/Alapan-Khan/Resume-Matcher.git
cd Resume-Matcher
pip install -r requirements.txt
streamlit run app.py


