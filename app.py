import streamlit as st
import os
import pandas as pd
from parser import parse_file
from extractor import extract_all_facts
from scorer import score_cv, generate_feedback

# --- Page Config ---
st.set_page_config(page_title="FlashCV Pro (NLP Engine)", page_icon="📄", layout="wide")

# --- Resource Loading ---
import spacy
import spacy.cli

@st.cache_resource
def load_resources():
    try:
        nlp = spacy.load("en_core_web_md")
    except OSError:
        spacy.cli.download("en_core_web_md")
        nlp = spacy.load("en_core_web_md")
    return nlp

nlp = load_resources()

# --- App Header ---
st.title("📄 FlashCV Pro")
st.markdown("Advanced NLP-Driven Semantic CV Engine")
st.divider()

# --- Top Section: Inputs ---
col1, col2 = st.columns(2)

with col1:
    st.subheader("1. Job Description")
    jd_text = st.text_area("Paste the requirements here:", height=300, placeholder="Enter job description...")

with col2:
    st.subheader("2. Upload CVs")
    uploaded_files = st.file_uploader("Upload PDF, DOCX, or Image files:", type=["pdf", "docx", "png", "jpg", "jpeg"], accept_multiple_files=True)

# --- Session State ---
if "analysis_results" not in st.session_state:
    st.session_state.analysis_results = None

# --- Controls ---
st.write("---")
c1, c2, c3 = st.columns([2, 1, 1])
with c1:
    if st.button("Analyze and Rank Candidates", type="primary", use_container_width=True):
        if not jd_text.strip():
            st.warning("Please provide a Job Description.")
        elif not uploaded_files:
            st.warning("Please upload at least one CV.")
        else:
            results = []
            with st.spinner("Running Semantic NLP Analysis..."):
                for uploaded_file in uploaded_files:
                    cv_text = parse_file(uploaded_file, uploaded_file.name)
                    if cv_text.strip():
                        facts = extract_all_facts(cv_text, nlp, jd_text)
                        scoring = score_cv(facts, cv_text, jd_text, nlp)
                        feedback = generate_feedback(facts, scoring)
                        results.append({
                            "filename": uploaded_file.name,
                            "cv_text": cv_text,
                            "facts": facts,
                            "scoring": scoring,
                            "feedback": feedback,
                            "score": scoring["overall_score"]
                        })
            if results:
                results.sort(key=lambda x: x["score"], reverse=True)
                st.session_state.analysis_results = results
                st.success(f"Analysis complete for {len(results)} candidate(s).")

with c2:
    if st.button("Load Demo Data", use_container_width=True):
        st.session_state.analysis_results = [
            {
                "filename": "Aaryan_Dev.pdf",
                "score": 85,
                "cv_text": "Aaryan Dev\nExperience: 6 Years\nConcepts: Python, Streamlit, Machine Learning...",
                "facts": {"email": "aaryan@dev.com", "phone": "+92 300 1234567", "experience": {"total_years": 6, "contextual_years": 6}, "skills": ["python", "streamlit", "machine learning", "data pipelines"]},
                "scoring": {
                    "overall_score": 85,
                    "category_scores": {"Smart Skill Match": 85, "Contextual Experience": 100, "Metrics & Structure": 90, "Action Verbs": 85, "Extra Skills": 0},
                    "matching_skills": ["python", "streamlit"],
                    "missing_skills": ["docker containerization", "ci/cd"],
                    "required_skills": [],
                    "is_unqualified": False
                },
                "feedback": ["Consider adding metrics to your recent role."]
            }
        ]

with c3:
    if st.button("Clear Results", use_container_width=True):
        st.session_state.analysis_results = None
        st.rerun()

# --- Interactive Leaderboard ---
if st.session_state.analysis_results:
    st.subheader("🏆 Candidate Leaderboard")
    
    for i, res in enumerate(st.session_state.analysis_results):
        rank = i + 1
        score = res["score"]
        name = res["filename"]
        email = res["facts"]["email"] or "N/A"
        
        status_icon = "🚨" if res["scoring"].get("is_unqualified") else "✅"
        
        with st.expander(f"{status_icon} Rank {rank} | {name} | Score: {score}% | {email}", expanded=False):
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Score Overview", "Semantic Alignment", "Advanced Insights", "Suggestions", "Raw Text"])
            
            with tab1:
                det_col1, det_col2 = st.columns(2)
                
                with det_col1:
                    if res["scoring"].get("is_unqualified"):
                        st.error("🚨 UNQUALIFIED: <30% Semantic Match. Scoring Halted.")
                        
                    st.metric("Hybrid Match Score", f"{score}%")
                    st.progress(max(0.0, min(1.0, score / 100.0)))
                    st.write("**Candidate Details:**")
                    st.write(f"- 📧 Email: {email}")
                    st.write(f"- 📞 Phone: {res['facts']['phone'] or 'Not Found'}")
                    
                    exp = res['facts'].get('experience', {})
                    if isinstance(exp, dict):
                        st.write(f"- ⏱️ Total Exp: {exp.get('total_years', 0)} years")
                        st.write(f"- 🎯 Contextual Exp: {exp.get('contextual_years', 0)} years")
                    else:
                        st.write(f"- ⏱️ Experience: {exp or 'Not Found'}")
                
                with det_col2:
                    st.write("**Category Breakdown:**")
                    for cat, val in res["scoring"]["category_scores"].items():
                        st.write(f"{cat}")
                        st.progress(max(0.0, min(1.0, val / 100.0)))
            
            with tab2:
                sk_col1, sk_col2 = st.columns(2)
                with sk_col1:
                    st.write("**Matched Semantic Concepts:**")
                    if res["scoring"]["matching_skills"]:
                        for s in res["scoring"]["matching_skills"]:
                            st.write(f"✅ {s}")
                    else:
                        st.write("No matching concepts found.")
                with sk_col2:
                    st.write("**Missing Semantic Concepts:**")
                    if res["scoring"]["missing_skills"]:
                        for s in res["scoring"]["missing_skills"]:
                            st.write(f"❌ {s}")
                    else:
                        st.write("No missing concepts!")
            
            with tab3:
                m1, m2, m3 = st.columns(3)
                exp_q = res["facts"].get("experience_quality", {})
                sec = res["facts"].get("sections", {}).get("sections_found", [])
                
                m1.metric("Action Verbs", len(exp_q.get("action_verbs", [])))
                m2.metric("Metrics Found", exp_q.get("metrics_count", 0))
                m3.metric("Sections Detected", len(sec))
                
                st.info(f"**Detected Sections:** {', '.join(sec) if sec else 'Standard structure detected.'}")
            
            with tab4:
                if res["feedback"]:
                    for fb in res["feedback"]:
                        st.warning(fb)
                else:
                    st.success("No improvement suggestions for this candidate.")
            
            with tab5:
                st.subheader("🔍 Extracted Raw Text")
                st.code(res.get("cv_text", "No text available."))
