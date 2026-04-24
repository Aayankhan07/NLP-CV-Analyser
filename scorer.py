def score_cv(cv_facts: dict, cv_text: str, jd_text: str, nlp) -> dict:
    scores = {}
    suggestions = []
    
    cv_doc = nlp(cv_text)
    jd_doc = nlp(jd_text)
    
    # 1. Semantic Relevance
    # NLP word vectors have a very high baseline because all English documents share generic words.
    # To fix this, we ONLY compute similarity on the core concepts (Noun Chunks) without stop words.
    cv_filtered_text = " ".join([chunk.lemma_.lower() for chunk in cv_doc.noun_chunks if not chunk.root.is_stop])
    jd_filtered_text = " ".join([chunk.lemma_.lower() for chunk in jd_doc.noun_chunks if not chunk.root.is_stop])
    
    cv_filt_doc = nlp(cv_filtered_text)
    jd_filt_doc = nlp(jd_filtered_text)

    if cv_filt_doc.has_vector and jd_filt_doc.has_vector:
        base_similarity = cv_filt_doc.similarity(jd_filt_doc)
        
        # Scale the similarity: map the realistic range [0.55, 1.0] to [0.0, 100.0]
        # This heavily penalizes completely unrelated domains (like Fashion Designer vs AI Engineer)
        adjusted_sim = max(0.0, (base_similarity - 0.55) / 0.45)
        relevance_score = adjusted_sim * 100.0
    else:
        relevance_score = 0.0
        suggestions.append("Could not compute semantic similarity due to missing word vectors.")
        
    scores["Semantic Relevance"] = round(relevance_score, 2)
    
    # Compare Semantic Concepts for Feedback
    jd_concepts = set([chunk.lemma_.lower() for chunk in jd_doc.noun_chunks if not chunk.root.is_stop])
    cv_concepts = set(cv_facts.get("skills", []))
    
    matching_concepts = jd_concepts.intersection(cv_concepts)
    missing_concepts = jd_concepts - cv_concepts
    
    if len(missing_concepts) > 0:
        suggestions.append(f"Missing Concepts: Consider incorporating these themes: {', '.join(list(missing_concepts)[:5])}")
    
    # 2. CV Structure & Formatting
    sections = cv_facts.get("sections", {}).get("sections_found", [])
    structure_score = min(100, len(sections) * 15) 
    scores["structure"] = structure_score
    if len(sections) < 3:
        suggestions.append("Formatting Fix: Your CV seems to lack clear, distinguishable section headers.")
        
    # 3. Work Experience Quality
    exp_quality = cv_facts.get("experience_quality", {})
    action_verbs = exp_quality.get("action_verbs", [])
    metrics_count = exp_quality.get("metrics_count", 0)
    
    action_verb_score = min(100, len(action_verbs) * 10)
    scores["action_verbs"] = action_verb_score
    if action_verb_score < 50:
        suggestions.append("Weak Sentences: Use more strong action verbs.")
        
    metrics_score = min(100, metrics_count * 20)
    scores["metrics"] = metrics_score
    if metrics_score < 50:
        suggestions.append("Measurable Achievements: Add more numbers or percentages to quantify your impact.")
        
    # 4. Consistency & Timeline
    timeline = cv_facts.get("timeline", {})
    gaps = timeline.get("career_gaps_detected", False)
    scores["timeline"] = 50 if gaps else 100
    if gaps:
        suggestions.append("Career Timeline: We detected potential gaps in your timeline.")

    weak_phrases = exp_quality.get("weak_phrases", [])
    grammar_score = max(0, 100 - (len(weak_phrases) * 15))
    scores["language"] = grammar_score
    if weak_phrases:
        suggestions.append(f"Improve Phrasing: You have {len(weak_phrases)} passive phrasing issues.")

    # Weighted Semantic Score
    weights = {
        "Semantic Relevance": 0.50,
        "structure": 0.10,
        "action_verbs": 0.15,
        "metrics": 0.15,
        "timeline": 0.05,
        "language": 0.05
    }
    
    overall_score = sum([scores[k] * w for k, w in weights.items()])
    
    # Relevance Gate: Good grammar shouldn't make an unrelated CV score highly
    if relevance_score < 30.0:
        overall_score = min(overall_score, relevance_score + 15.0)
        suggestions.insert(0, "CRITICAL: This CV appears to be from a completely different domain than the Job Description.")
    
    return {
        "overall_score": round(overall_score, 2),
        "category_scores": scores,
        "suggestions": suggestions,
        "matching_skills": list(matching_concepts),
        "missing_skills": list(missing_concepts),
        "required_skills": list(jd_concepts)
    }

def generate_feedback(extracted_facts: dict, scoring_results: dict) -> list:
    feedback = []
    if not extracted_facts.get("email"):
        feedback.append("Contact Completeness: No email address found on the CV.")
    if not extracted_facts.get("phone"):
        feedback.append("Contact Completeness: No phone number found on the CV.")
    feedback.extend(scoring_results.get("suggestions", []))
    return list(set(feedback))
