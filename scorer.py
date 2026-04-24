def score_cv(cv_facts: dict, cv_text: str, jd_text: str, nlp) -> dict:
    scores = {}
    suggestions = []
    
    cv_doc = nlp(cv_text)
    jd_doc = nlp(jd_text)
    
    # Phase 1: The Hard Gate
    cv_filtered_text = " ".join([chunk.lemma_.lower() for chunk in cv_doc.noun_chunks if not chunk.root.is_stop])
    jd_filtered_text = " ".join([chunk.lemma_.lower() for chunk in jd_doc.noun_chunks if not chunk.root.is_stop])
    
    cv_filt_doc = nlp(cv_filtered_text)
    jd_filt_doc = nlp(jd_filtered_text)

    if cv_filt_doc.has_vector and jd_filt_doc.has_vector:
        base_similarity = cv_filt_doc.similarity(jd_filt_doc)
        # Apply exponential penalty: ((Similarity - 0.65) / 0.35)^3 * 100
        adjusted_sim = max(0.0, (base_similarity - 0.65) / 0.35)
        relevance_score = (adjusted_sim ** 3) * 100.0
    else:
        relevance_score = 0.0
        suggestions.append("Could not compute semantic similarity due to missing word vectors.")
        
    if relevance_score < 30.0:
        # Halt Phase 2
        overall_score = min(35.0, relevance_score + 5.0)
        suggestions.insert(0, "CRITICAL ALERT: This CV has <30% semantic overlap with the Job Description. Flagged as UNQUALIFIED.")
        return {
            "overall_score": round(overall_score, 2),
            "category_scores": {"Semantic Relevance": round(relevance_score, 2)},
            "suggestions": suggestions,
            "matching_skills": [],
            "missing_skills": [],
            "required_skills": [],
            "is_unqualified": True
        }
        
    # Phase 2: The Hybrid Formula
    
    # 1. Smart Skill Match (50% Weight)
    scores["Smart Skill Match"] = round(relevance_score, 2)
    
    jd_concepts = set([chunk.lemma_.lower() for chunk in jd_doc.noun_chunks if not chunk.root.is_stop])
    cv_concepts = set(cv_facts.get("skills", []))
    matching_concepts = jd_concepts.intersection(cv_concepts)
    missing_concepts = jd_concepts - cv_concepts
    extra_concepts = cv_concepts - jd_concepts
    
    if len(missing_concepts) > 0:
        suggestions.append(f"Missing Concepts: Consider incorporating these themes: {', '.join(list(missing_concepts)[:5])}")

    # 2. Contextual Experience (30% Weight)
    exp_info = cv_facts.get("experience", {})
    contextual_years = exp_info.get("contextual_years", 0)
    jd_req = cv_facts.get("jd_requirements", {})
    required_years = jd_req.get("required_years", 2)
    
    if required_years > 0:
        exp_score = min(100.0, (contextual_years / required_years) * 100.0)
    else:
        exp_score = 100.0 if contextual_years > 0 else 0.0
        
    scores["Contextual Experience"] = round(exp_score, 2)
    if contextual_years < required_years:
        suggestions.append(f"Experience Gap: The JD requires {required_years} years, but we only found {contextual_years} years of semantically relevant experience.")

    # 3. Professional Quality & Extras (20% Weight)
    # Metrics & Structure (10%)
    sections = cv_facts.get("sections", {}).get("sections_found", [])
    structure_score = min(100, len(sections) * 15) 
    
    exp_quality = cv_facts.get("experience_quality", {})
    metrics_count = exp_quality.get("metrics_count", 0)
    metrics_score = min(100, metrics_count * 20)
    
    metrics_structure_score = (structure_score + metrics_score) / 2
    scores["Metrics & Structure"] = round(metrics_structure_score, 2)
    
    if len(sections) < 3:
        suggestions.append("Formatting Fix: Your CV seems to lack clear, distinguishable section headers.")
    if metrics_score < 50:
        suggestions.append("Measurable Achievements: Add more numbers or percentages to quantify your impact.")

    # Action Verbs (5%)
    action_verbs = exp_quality.get("action_verbs", [])
    action_verb_score = min(100, len(action_verbs) * 10)
    
    weak_phrases = exp_quality.get("weak_phrases", [])
    if weak_phrases:
        action_verb_score = max(0, action_verb_score - (len(weak_phrases) * 10))
        suggestions.append(f"Improve Phrasing: You have {len(weak_phrases)} passive phrasing issues.")
        
    timeline = cv_facts.get("timeline", {})
    if timeline.get("career_gaps_detected", False):
        action_verb_score = max(0, action_verb_score - 20)
        suggestions.append("Career Timeline: We detected potential gaps in your timeline.")
        
    scores["Action Verbs"] = round(action_verb_score, 2)
    if action_verb_score < 50:
        suggestions.append("Weak Sentences: Use more strong action verbs and fix passive phrasing.")

    # Extra Skills (5%)
    extra_score = min(100, len(extra_concepts) * 5)
    scores["Extra Skills"] = round(extra_score, 2)
    
    # Hybrid Calculation
    weights = {
        "Smart Skill Match": 0.50,
        "Contextual Experience": 0.30,
        "Metrics & Structure": 0.10,
        "Action Verbs": 0.05,
        "Extra Skills": 0.05
    }
    
    overall_score = sum([scores[k] * w for k, w in weights.items()])
    
    return {
        "overall_score": round(overall_score, 2),
        "category_scores": scores,
        "suggestions": suggestions,
        "matching_skills": list(matching_concepts),
        "missing_skills": list(missing_concepts),
        "required_skills": list(jd_concepts),
        "is_unqualified": False
    }

def generate_feedback(extracted_facts: dict, scoring_results: dict) -> list:
    feedback = []
    if not extracted_facts.get("email"):
        feedback.append("Contact Completeness: No email address found on the CV.")
    if not extracted_facts.get("phone"):
        feedback.append("Contact Completeness: No phone number found on the CV.")
    feedback.extend(scoring_results.get("suggestions", []))
    return list(set(feedback))
