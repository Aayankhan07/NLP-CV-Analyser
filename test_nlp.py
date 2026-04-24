import sys
import os
import spacy

from extractor import extract_experience, analyze_experience, analyze_timeline, extract_all_facts
from parser import parse_pdf
from flashtext import KeywordProcessor

def test_nlp():
    print("Loading spaCy...")
    nlp = spacy.load("en_core_web_sm")
    print("spaCy loaded.")
    
    sample_text = """
    Aaryan Dev
    Email: aaryan@dev.com
    Phone: +92 300 1234567
    
    Experience
    Software Engineer at TechCorp (2018 - 2024)
    I managed a team of developers and optimized the database queries resulting in 30% performance boost and $50k cost savings.
    I have 6 years of experience in Python and Streamlit.
    """
    
    print("Extracting experience...")
    exp = extract_experience(sample_text, nlp)
    print("Experience:", exp)
    
    print("Analyzing experience quality...")
    quality = analyze_experience(sample_text, None, nlp)
    print("Action verbs:", quality["action_verbs"])
    print("Metrics count:", quality["metrics_count"])
    
    print("Analyzing timeline...")
    timeline = analyze_timeline(sample_text, nlp)
    print("Timeline:", timeline)

if __name__ == "__main__":
    test_nlp()
