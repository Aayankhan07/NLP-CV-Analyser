import re
import json
from flashtext import KeywordProcessor

def load_taxonomy(path="taxonomy.json"):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return {}

def extract_email(text: str) -> str:
    """
    Extracts the first email address found in the text using a regular expression.
    """
    email_pattern = r'[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}'
    match = re.search(email_pattern, text)
    return match.group(0) if match else None

def extract_phone(text: str) -> str:
    """
    Extracts the first phone number found in the text using a regular expression.
    This pattern looks for international or domestic numbers with varying separators.
    """
    # A generic phone pattern: handles optional +, country codes, parentheses, and dashes/spaces
    phone_pattern = r'(\+?\d{1,3}[-.\s]?)?(\(?\d{3}\)?[-.\s]?)?\d{3}[-.\s]?\d{4}'
    match = re.search(phone_pattern, text)
    if match:
        # Check if the extracted string has enough digits to be a real phone number
        digits = re.sub(r'\D', '', match.group(0))
        if len(digits) >= 10:
            return match.group(0)
    return None

def extract_experience(text: str, nlp) -> str:
    """
    Extracts total years of experience using spaCy NLP (Dependency Parsing / POS).
    Looks for numerical tokens linked to 'year' or 'yr'.
    """
    doc = nlp(text)
    years = []
    
    for token in doc:
        if token.lemma_.lower() in ["year", "yr", "years", "yrs"]:
            # Check direct children for numbers
            for child in token.children:
                if child.pos_ == "NUM" or child.like_num:
                    try:
                        years.append(int(child.text))
                    except ValueError:
                        pass
            
            # Check previous token if child didn't catch it
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

def get_keyword_processor(skills_json_path: str = "skills.json") -> KeywordProcessor:
    """
    Initializes a FlashText KeywordProcessor with skills from a JSON file.
    """
    keyword_processor = KeywordProcessor()
    
    try:
        with open(skills_json_path, 'r', encoding='utf-8') as f:
            skills_dict = json.load(f)
            
        # flashtext add_keywords_from_dict expects the format:
        # {"Skill Name": ["synonym1", "synonym2", ...]}
        keyword_processor.add_keywords_from_dict(skills_dict)
    except Exception as e:
        print(f"Error loading {skills_json_path}: {e}")
        
    return keyword_processor

def extract_skills(text: str, keyword_processor: KeywordProcessor) -> set:
    """
    Extracts skills from text using FlashText and returns a unique set.
    """
    # Extract keywords; FlashText will return the standardized key (e.g., "Python")
    found_skills = keyword_processor.extract_keywords(text)
    return set(found_skills)

def extract_sections(text: str, taxonomy: dict) -> dict:
    sections_found = set()
    if not taxonomy: return {"sections_found": []}
    for section, headers in taxonomy.get("section_headers", {}).items():
        for header in headers:
            if re.search(r'\b' + re.escape(header) + r'\b', text, re.IGNORECASE):
                sections_found.add(section)
                break
    return {"sections_found": list(sections_found)}

def analyze_experience(text: str, taxonomy: dict, nlp) -> dict:
    action_verbs_found = set()
    weak_phrases_found = set()
    
    doc = nlp(text)
    
    # Use NLP POS tagging to find verbs dynamically instead of a hardcoded taxonomy list
    for token in doc:
        if token.pos_ == "VERB" and token.is_alpha and len(token.text) > 2:
            action_verbs_found.add(token.lemma_.lower())
            
        # Detect passive voice as a "weak phrase" using dependency parsing
        if token.dep_ == "auxpass":
            weak_phrases_found.add(f"Passive voice: '{token.head.text}'")
            
    # Fallback to taxonomy for specific weak phrases if they exist
    if taxonomy:
        for phrase in taxonomy.get("weak_phrases", []):
            if re.search(r'\b' + re.escape(phrase) + r'\b', text, re.IGNORECASE):
                weak_phrases_found.add(phrase)
                
    # Detect metrics (percentages, currencies, multipliers)
    metrics = re.findall(r'(\$\d+[mMkK]?|\d+%|\d+x)', text, re.IGNORECASE)
    
    # Cap the list size
    action_verbs_list = list(action_verbs_found)[:50]
    
    return {
        "action_verbs": action_verbs_list,
        "weak_phrases": list(weak_phrases_found),
        "metrics_count": len(metrics)
    }

def analyze_timeline(text: str, nlp) -> dict:
    doc = nlp(text)
    years = set()
    
    # Extract years dynamically using Named Entity Recognition for DATE labels
    for ent in doc.ents:
        if ent.label_ == "DATE":
            # Extract possible years from the date entity
            extracted_years = re.findall(r'\b(19\d{2}|20\d{2})\b', ent.text)
            for y in extracted_years:
                years.add(int(y))

    years = sorted(list(years))
    gaps = False
    if len(years) > 1:
        # Check if max gap > 2 years
        for i in range(1, len(years)):
            if years[i] - years[i-1] > 2:
                gaps = True
                break
    return {"career_gaps_detected": gaps, "years_found": years}

def extract_all_facts(text: str, keyword_processor: KeywordProcessor, taxonomy: dict = None, nlp=None) -> dict:
    """
    Extracts all facts from the CV text using hybrid NLP and rules.
    """
    return {
        "email": extract_email(text),
        "phone": extract_phone(text),
        "experience": extract_experience(text, nlp),
        "skills": extract_skills(text, keyword_processor),
        "sections": extract_sections(text, taxonomy),
        "experience_quality": analyze_experience(text, taxonomy, nlp),
        "timeline": analyze_timeline(text, nlp)
    }
