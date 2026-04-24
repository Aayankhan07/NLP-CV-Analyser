import sys
import os
import spacy

from extractor import extract_experience, analyze_experience, analyze_timeline, extract_all_facts
from scorer import score_cv

def test_nlp():
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_md")
    print("spaCy loaded.")
    
    sample_text = """
    Aaryan Dev
    Email: aaryan@dev.com
    Phone: +92 300 1234567
    
    WORK EXPERIENCE
    Software Engineer at TechCorp (2018 - 2024)
    I managed a team of developers and optimized the database queries resulting in 30% performance boost and $50k cost savings.
    I have 6 years of experience in Python and Streamlit.
    """
    
    jd_text = """
    We are looking for a Software Engineer with experience in Python, web development, and database optimization.
    Must have strong management skills and be able to deliver performance boosts.
    """
    
    print("Extracting all facts...")
    facts = extract_all_facts(sample_text, nlp, jd_text)
    print("Facts extracted:")
    for k, v in facts.items():
        print(f" - {k}: {v}")
        
    print("\nScoring CV against JD...")
    scoring = score_cv(facts, sample_text, jd_text, nlp)
    print("Overall Score:", scoring["overall_score"])
    print("Semantic Relevance:", scoring["category_scores"]["Semantic Relevance"])
    print("Missing Concepts:", scoring["missing_skills"])
    print("Suggestions:", scoring["suggestions"])

if __name__ == "__main__":
    test_nlp()
