import streamlit as st
import os
import pandas as pd
from parser import parse_file
from extractor import get_keyword_processor, extract_all_facts, load_taxonomy
from scorer import score_cv, generate_feedback

# --- Page Config ---
st.set_page_config(page_title="FlashCV Pro", page_icon="📄", layout="wide")

# --- Resource Loading ---
import spacy

@st.cache_resource
def load_resources():
    kp = get_keyword_processor("skills.json") if os.path.exists("skills.json") else None
    tax = load_taxonomy("taxonomy.json") if os.path.exists("taxonomy.json") else {}
    
    try:
        nlp = spacy.load("en_core_web_sm")
    except OSError:
        import spacy.cli
        spacy.cli.download("en_core_web_sm")
        nlp = spacy.load("en_core_web_sm")
        
    return kp, tax, nlp

keyword_processor, taxonomy, nlp = load_resources()

# --- App Header ---
st.title("📄 FlashCV Pro")
st.markdown("Advanced Deterministic CV Intelligence & Ranking Engine")
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
            with st.spinner("Analyzing CVs..."):
                for uploaded_file in uploaded_files:
                    cv_text = parse_file(uploaded_file, uploaded_file.name)
                    if cv_text.strip():
                        facts = extract_all_facts(cv_text, keyword_processor, taxonomy, nlp)
                        scoring = score_cv(facts, jd_text, keyword_processor)
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
                "score": 92,
                "cv_text": "Aaryan Dev\nExperience: 6 Years\nSkills: Python, Streamlit, SQL...",
                "facts": {"email": "aaryan@dev.com", "phone": "+92 300 1234567", "experience": "6 Years", "skills": {"Python", "Streamlit", "SQL"}},
                "scoring": {
                    "overall_score": 92,
                    "category_scores": {"Relevance": 95, "Structure": 90, "Action_Verbs": 85, "Metrics": 80},
                    "matching_skills": ["Python", "Streamlit"],
                    "missing_skills": ["Docker"],
                    "required_skills": {"Python", "Streamlit", "Docker"}
                },
                "feedback": ["Great resume.", "Consider adding Docker projects."]
            },
            {
                "filename": "Sample_CV_02.docx",
                "score": 68,
                "cv_text": "Sample Candidate\nExperience: 2 Years\nSkills: Python, HTML...",
                "facts": {"email": "sample@test.com", "phone": "N/A", "experience": "2 Years", "skills": {"Python", "HTML"}},
                "scoring": {
                    "overall_score": 68,
                    "category_scores": {"Relevance": 60, "Structure": 75, "Action_Verbs": 70, "Metrics": 50},
                    "matching_skills": ["Python"],
                    "missing_skills": ["Streamlit", "SQL"],
                    "required_skills": {"Python", "Streamlit", "SQL"}
                },
                "feedback": ["Need more focus on data tools.", "Quantify your achievements."]
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
        
        # Expander acts as the interactive leaderboard row
        with st.expander(f"Rank {rank} | {name} | Score: {score}% | {email}", expanded=False):
            
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Score Overview", "Skill Alignment", "Advanced Insights", "Suggestions", "Raw Text"])
            
            with tab1:
                det_col1, det_col2 = st.columns(2)
                
                with det_col1:
                    st.metric("Overall Match Score", f"{score}%")
                    st.progress(score / 100.0)
                    st.write("**Candidate Details:**")
                    st.write(f"- 📧 Email: {email}")
                    st.write(f"- 📞 Phone: {res['facts']['phone'] or 'Not Found'}")
                    st.write(f"- ⏱️ Experience: {res['facts']['experience'] or 'Not Found'}")
                
                with det_col2:
                    st.write("**Category Breakdown:**")
                    for cat, val in res["scoring"]["category_scores"].items():
                        st.write(f"{cat.replace('_', ' ').title()}")
                        st.progress(val / 100.0)
            
            with tab2:
                sk_col1, sk_col2 = st.columns(2)
                with sk_col1:
                    st.write("**Matching Skills:**")
                    if res["scoring"]["matching_skills"]:
                        for s in res["scoring"]["matching_skills"]:
                            st.write(f"✅ {s}")
                    else:
                        st.write("No matches found.")
                with sk_col2:
                    st.write("**Missing Skills:**")
                    if res["scoring"]["missing_skills"]:
                        for s in res["scoring"]["missing_skills"]:
                            st.write(f"❌ {s}")
                    else:
                        st.write("No missing skills!")
            
            with tab3:
                m1, m2, m3 = st.columns(3)
                exp_q = res["facts"].get("experience_quality", {})
                sec = res["facts"].get("sections", {}).get("sections_found", [])
                
                m1.metric("Action Verbs", len(exp_q.get("action_verbs", [])))
                m2.metric("Metrics Found", exp_q.get("metrics_count", 0))
                m3.metric("Sections", len(sec))
                
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
