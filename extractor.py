import spacy
from spacy.matcher import Matcher

def extract_email(nlp, doc):
    for token in doc:
        if token.like_email:
            return token.text
    return None

def extract_phone(nlp, doc):
    # Pure NLP contextual parsing: Look for tokens that are long numbers, 
    # then look backwards for country code indicators like (+91)
    for i, token in enumerate(doc):
        # Identify the core of the phone number (7+ digits)
        if token.is_digit and len(token.text) >= 7:
            start = i
            # Look backwards up to 3 tokens for country codes and brackets
            for j in range(i - 1, max(-1, i - 4), -1):
                prev_token = doc[j]
                if prev_token.text in ["(", ")", "-", "+"] or prev_token.text.startswith("+"):
                    start = j
                else:
                    break
            return doc[start:i+1].text.strip()
    return None

def extract_experience(nlp, doc, jd_doc=None):
    blocks = doc.text.split('\n')
    contextual_years = 0
    total_years = 0
    
    for block in blocks:
        if not block.strip():
            continue
            
        block_doc = nlp(block)
        block_years = []
        for token in block_doc:
            if token.lemma_.lower() in ["year", "yr", "years", "yrs"]:
                for child in token.children:
                    if child.pos_ == "NUM" or child.like_num:
                        try:
                            block_years.append(int(child.text))
                        except ValueError:
                            pass
                if token.i > 0:
                    prev_token = block_doc[token.i - 1]
                    if prev_token.pos_ == "NUM" or prev_token.like_num:
                        try:
                            block_years.append(int(prev_token.text))
                        except ValueError:
                            pass
                            
        if block_years:
            years = max(block_years)
            # If the block has a crazy number, ignore it
            if years > 50:
                continue
                
            total_years += years
            
            # Contextual validation
            if jd_doc:
                jd_filt = nlp(" ".join([t.lemma_.lower() for t in jd_doc.noun_chunks if not t.root.is_stop]))
                block_filt = nlp(" ".join([t.lemma_.lower() for t in block_doc.noun_chunks if not t.root.is_stop]))
                
                if block_filt.has_vector and jd_filt.has_vector and len(block_filt) > 0:
                    sim = block_filt.similarity(jd_filt)
                    if sim > 0.40: # Semantic threshold for a block to be "relevant"
                        contextual_years += years
            else:
                contextual_years += years

    return {
        "total_years": total_years,
        "contextual_years": contextual_years
    }


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

def extract_jd_requirements(jd_text, nlp):
    jd_doc = nlp(jd_text)
    req_years = 2 # Default to 2 if no explicit requirement found
    
    for token in jd_doc:
        if token.lemma_.lower() in ["year", "yr", "years", "yrs"]:
            # Check children
            for child in token.children:
                if child.pos_ == "NUM" or child.like_num:
                    try:
                        req_years = max(req_years, int(child.text))
                    except ValueError:
                        pass
            # Check previous token
            if token.i > 0:
                prev_token = jd_doc[token.i - 1]
                if prev_token.pos_ == "NUM" or prev_token.like_num:
                    try:
                        req_years = max(req_years, int(prev_token.text))
                    except ValueError:
                        pass
    return {"required_years": req_years}

def extract_all_facts(text: str, nlp, jd_text: str = None) -> dict:
    doc = nlp(text)
    jd_doc = nlp(jd_text) if jd_text else None
    
    facts = {
        "email": extract_email(nlp, doc),
        "phone": extract_phone(nlp, doc),
        "experience": extract_experience(nlp, doc, jd_doc),
        "skills": extract_skills(nlp, doc),
        "sections": extract_sections(nlp, doc),
        "experience_quality": analyze_experience(nlp, doc),
        "timeline": analyze_timeline(nlp, doc)
    }
    
    if jd_doc:
        facts["jd_requirements"] = extract_jd_requirements(jd_text, nlp)
        
    return facts
