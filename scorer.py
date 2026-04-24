from flashtext import KeywordProcessor

def score_cv(cv_facts: dict, jd_text: str, keyword_processor: KeywordProcessor) -> dict:
    """
    Advanced Scoring Engine computing scores across multiple categories.
    """
    scores = {}
    suggestions = []
    
    cv_skills = cv_facts.get("skills", set())
    
    # 1. Relevance & Skills Matching
    extracted_jd_keywords = keyword_processor.extract_keywords(jd_text.lower())
    required_skills = set(extracted_jd_keywords)
    
    missing_skills = required_skills - cv_skills
    matching_skills = required_skills.intersection(cv_skills)
    
    if required_skills:
        relevance_score = (len(matching_skills) / len(required_skills)) * 100.0
    else:
        relevance_score = 0.0
        suggestions.append("No recognizable skills found in the Job Description.")
        
    scores["relevance"] = round(relevance_score, 2)
    
    if len(missing_skills) > 0:
        suggestions.append(f"Missing Keywords: Consider adding these required skills: {', '.join(list(missing_skills)[:5])}")
    
    # 2. CV Structure & Formatting
    sections = cv_facts.get("sections", {}).get("sections_found", [])
    structure_score = min(100, len(sections) * 20) # 20 points per standard section
    scores["structure"] = structure_score
    
    if "experience" not in sections:
        suggestions.append("Formatting Fix: Your CV is missing a clear 'Experience' or 'Work History' heading.")
    if "education" not in sections:
        suggestions.append("Formatting Fix: Your CV is missing an 'Education' heading.")
        
    # 3. Work Experience Quality & Use of Action Verbs
    exp_quality = cv_facts.get("experience_quality", {})
    action_verbs = exp_quality.get("action_verbs", [])
    metrics_count = exp_quality.get("metrics_count", 0)
    
    # Action verbs score (up to 100)
    action_verb_score = min(100, len(action_verbs) * 10)
    scores["action_verbs"] = action_verb_score
    if action_verb_score < 50:
        suggestions.append("Weak Sentences: Use more strong action verbs (e.g., 'managed', 'developed', 'optimized').")
        
    # Metrics score
    metrics_score = min(100, metrics_count * 20)
    scores["metrics"] = metrics_score
    if metrics_score < 50:
        suggestions.append("Measurable Achievements: Add more numbers, percentages (%), or dollar amounts ($) to quantify your impact.")
        
    # 4. Consistency & Timeline
    timeline = cv_facts.get("timeline", {})
    gaps = timeline.get("career_gaps_detected", False)
    scores["timeline"] = 50 if gaps else 100
    if gaps:
        suggestions.append("Career Timeline: We detected potential gaps in your timeline (more than 2 years between dates). Ensure this is explained or formatted clearly.")

    # 5. Language & Grammar (Heuristic based on weak phrases)
    weak_phrases = exp_quality.get("weak_phrases", [])
    grammar_score = max(0, 100 - (len(weak_phrases) * 15))
    scores["language"] = grammar_score
    if weak_phrases:
        suggestions.append(f"Improve Phrasing: Avoid passive or generic phrases like: '{weak_phrases[0]}'.")

    # Overall ATS Score (Weighted Average)
    weights = {
        "relevance": 0.40,
        "structure": 0.15,
        "action_verbs": 0.15,
        "metrics": 0.15,
        "timeline": 0.05,
        "language": 0.10
    }
    
    overall_score = sum([scores[k] * w for k, w in weights.items()])
    
    return {
        "overall_score": round(overall_score, 2),
        "category_scores": scores,
        "suggestions": suggestions,
        "matching_skills": matching_skills,
        "missing_skills": missing_skills,
        "required_skills": required_skills
    }

def generate_feedback(extracted_facts: dict, scoring_results: dict) -> list:
    """
    Combines fact-based feedback with scoring-based suggestions.
    """
    feedback = []
    
    if not extracted_facts.get("email"):
        feedback.append("Contact Completeness: No email address found on the CV.")
        
    if not extracted_facts.get("phone"):
        feedback.append("Contact Completeness: No phone number found on the CV.")
        
    # Add scoring suggestions
    feedback.extend(scoring_results.get("suggestions", []))
        
    return list(set(feedback)) # Remove duplicates
