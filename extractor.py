import spacy
from spacy.matcher import Matcher

def extract_email(nlp, doc):
    for token in doc:
        if token.like_email:
            return token.text
    return None

def extract_phone(nlp, doc):
    # Pure NLP pattern matching for phone numbers using spaCy Matcher
    # Removes regex. Looks for sequences of numbers and common punctuation.
    matcher = Matcher(nlp.vocab)
    pattern1 = [{"SHAPE": "ddd"}, {"TEXT": "-", "OP": "?"}, {"SHAPE": "ddd"}, {"TEXT": "-", "OP": "?"}, {"SHAPE": "dddd"}]
    pattern2 = [{"TEXT": "+"}, {"SHAPE": "dd"}, {"SHAPE": "ddd"}, {"SHAPE": "ddddddd"}] 
    matcher.add("PHONE", [pattern1, pattern2])
    matches = matcher(doc)
    if matches:
        match_id, start, end = matches[0]
        return doc[start:end].text
    return None

def extract_experience(nlp, doc):
    years = []
    for token in doc:
        if token.lemma_.lower() in ["year", "yr", "years", "yrs"]:
            for child in token.children:
                if child.pos_ == "NUM" or child.like_num:
                    try:
                        years.append(int(child.text))
                    except ValueError:
                        pass
            if token.i > 0:
                prev_token = doc[token.i - 1]
                if prev_token.pos_ == "NUM" or prev_token.like_num:
                    try:
                        years.append(int(prev_token.text))
                    except ValueError:
                        pass
    if years:
        return f"{max(years)} years"
    return None

def extract_skills(nlp, doc):
    # Extract noun chunks that aren't purely stop words as conceptual skills
    concepts = set()
    for chunk in doc.noun_chunks:
        if not chunk.root.is_stop and chunk.root.pos_ in ["NOUN", "PROPN"]:
            concepts.add(chunk.lemma_.lower().strip())
    return list(concepts)

def extract_sections(nlp, doc):
    # NLP approach to find headers: short noun phrases that are title-cased or fully uppercase
    sections = set()
    for chunk in doc.noun_chunks:
        if len(chunk) <= 3 and (chunk.text.isupper() or chunk.text.istitle()):
            sections.add(chunk.text.strip())
    return {"sections_found": list(sections)}

def analyze_experience(nlp, doc):
    action_verbs_found = set()
    weak_phrases_found = set()
    metrics_count = 0
    
    for token in doc:
        if token.pos_ == "VERB" and token.is_alpha and len(token.text) > 2:
            action_verbs_found.add(token.lemma_.lower())
        if token.dep_ == "auxpass":
            weak_phrases_found.add(f"Passive voice: '{token.head.text}'")
            
    # Use NER for metrics
    for ent in doc.ents:
        if ent.label_ in ["MONEY", "PERCENT", "CARDINAL"]:
            metrics_count += 1
            
    return {
        "action_verbs": list(action_verbs_found)[:50],
        "weak_phrases": list(weak_phrases_found),
        "metrics_count": metrics_count
    }

def analyze_timeline(nlp, doc):
    years = set()
    for ent in doc.ents:
        if ent.label_ == "DATE":
            # Just extract 4 digit tokens without regex
            for token in ent:
                if token.like_num and len(token.text) == 4:
                    try:
                        years.add(int(token.text))
                    except ValueError:
                        pass
    years = sorted(list(years))
    gaps = False
    if len(years) > 1:
        for i in range(1, len(years)):
            if years[i] - years[i-1] > 2:
                gaps = True
                break
    return {"career_gaps_detected": gaps, "years_found": years}

def extract_all_facts(text: str, nlp) -> dict:
    doc = nlp(text)
    return {
        "email": extract_email(nlp, doc),
        "phone": extract_phone(nlp, doc),
        "experience": extract_experience(nlp, doc),
        "skills": extract_skills(nlp, doc),
        "sections": extract_sections(nlp, doc),
        "experience_quality": analyze_experience(nlp, doc),
        "timeline": analyze_timeline(nlp, doc)
    }
