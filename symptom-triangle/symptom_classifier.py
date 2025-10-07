import re
import spacy

nlp = spacy.load("en_core_web_sm")

# Department keyword mapping
DEPARTMENT_KEYWORDS = {
    "Emergency": ["unconscious", "collapse", "trauma", "bleeding", "shock", "cyanosis", "seizure"],
    "Cardiology": ["chest pain", "palpitations", "pressure", "heartbeat", "sweating", "tightness", "fatigue"],
    "Pulmonology": ["breathing", "cough", "asthma", "wheezing", "apnea", "sputum"],
    "Neurology": ["numbness", "confusion", "paralysis", "stroke", "seizure", "memory loss", "slurred speech"],
    "Gastroenterology": ["vomiting", "diarrhea", "abdominal", "stomach", "heartburn", "constipation"],
    "Orthopedics": ["fracture", "joint", "bone", "limb", "stiffness", "swelling"],
    "Dermatology": ["rash", "itching", "redness", "blisters", "acne", "eczema"],
    "Psychiatry": ["anxiety", "depression", "mood", "panic", "insomnia", "hallucinations"],
    "ENT": ["ear", "throat", "nose", "sinus", "hearing", "tinnitus"],
    "Ophthalmology": ["vision", "eye", "blurred", "redness", "photophobia"],
    "Urology": ["urination", "bladder", "kidney", "pee", "incontinence"],
    "Gynecology": ["period", "pelvic", "pregnancy", "vaginal", "cramps"]
}

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-z\s]", "", text)
    doc = nlp(text)
    tokens = [token.lemma_ for token in doc if not token.is_stop]
    return tokens

def extract_symptom_phrases(text):
    doc = nlp(text)
    phrases = []
    for chunk in doc.noun_chunks:
        phrases.append(chunk.text.lower())
    return phrases

def classify_symptoms(text):
    phrases = extract_symptom_phrases(text)
    scores = {dept: 0 for dept in DEPARTMENT_KEYWORDS}
    for phrase in phrases:
        for dept, keywords in DEPARTMENT_KEYWORDS.items():
            for kw in keywords:
                if kw in phrase:
                    scores[dept] += 1
    sorted_departments = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [dept for dept, score in sorted_departments if score > 0]