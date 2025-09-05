import pytesseract
from PIL import Image
from difflib import SequenceMatcher
import re

# --- CONFIGURATION ---
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Weighted EPC keywords (more important → higher weight)
EPC_KEYWORDS = {
    "Primärenergie": 4,
    "CO2": 4,
    "Heizbedarf": 4,
    "Energie": 2,
    "Passivhaus": 2,
    "Wohngebäude": 1,
    "Referenzgebäude": 1,
    "Gebäudehülle": 2,
    "Fenster": 1,
    "Dämmung": 2
}

CRITICAL_KEYWORDS = ["Primärenergie", "CO2", "Heizbedarf"]

# --- UTILITY FUNCTIONS ---
def extract_text_from_image(image_path, lang='deu'):
    """Extract text from an image using OCR"""
    img = Image.open(image_path)
    text = pytesseract.image_to_string(img, lang=lang)
    return text

def clean_text(text):
    """Normalize text by lowercasing and removing extra spaces"""
    return re.sub(r'\s+', ' ', text.lower()).strip()

def compute_similarity(text1, text2):
    """Sequence similarity (difflib)"""
    return SequenceMatcher(None, text1, text2).ratio()

def keyword_score(text):
    """Compute weighted keyword score"""
    text_clean = clean_text(text)
    total_weight = sum(EPC_KEYWORDS.values())
    found_weight = sum(weight for kw, weight in EPC_KEYWORDS.items() if kw.lower() in text_clean)
    
    ratio = found_weight / total_weight if total_weight > 0 else 0.0
    
    # Bonus if multiple critical keywords are found
    critical_found = sum(1 for kw in CRITICAL_KEYWORDS if kw.lower() in text_clean)
    if critical_found >= 2:
        ratio = min(1.0, ratio + 0.2)
    
    return ratio

def structure_score(text):
    """
    Optional structural scoring:
    - Count tables, headers, bullet points, etc.
    - Reward organized EPC-like layout
    """
    score = 0.0
    # simple proxy: count bullet points or table-like patterns
    bullets = len(re.findall(r"[-•\*]", text))
    tables = len(re.findall(r"(?:\|\s*\w+\s*){2,}\|", text))  # simple Markdown table-like
    score += min(1.0, 0.1 * bullets + 0.1 * tables)
    return score

def classify_score(score, epc_threshold=0.88, non_epc_threshold=0.6):
    if score >= epc_threshold:
        return "EPC"
    elif score <= non_epc_threshold:
        return "Non-EPC"
    else:
        return "Indeterminate"

# --- MAIN SCORING FUNCTION ---
def score_document(reference_text, candidate_text, alpha=0.25, beta=0.15):
    """
    Final score combining:
    - Text similarity (alpha)
    - Keyword weighted score (1-alpha-beta)
    - Optional structure score (beta)
    """
    sim = compute_similarity(clean_text(reference_text), clean_text(candidate_text))
    key = keyword_score(candidate_text)
    struct = structure_score(candidate_text)
    
    final_score = alpha * sim + (1 - alpha - beta) * key + beta * struct
    final_score = max(0.0, min(1.0, final_score))
    
    classification = classify_score(final_score)
    return final_score, classification

# --- EXAMPLE USAGE ---
if __name__ == "__main__":
    reference_img = "reference.png"
    epc_img = "match.png"
    non_epc_img = "not_epc1.png"
    
    reference_text = extract_text_from_image(reference_img)
    epc_text = extract_text_from_image(epc_img)
    non_epc_text = extract_text_from_image(non_epc_img)
    
    epc_score, epc_class = score_document(reference_text, epc_text)
    non_epc_score, non_epc_class = score_document(reference_text, non_epc_text)
    
    print(f"EPC match score: {epc_score:.3f} → {epc_class}")
    print(f"Non-EPC match score: {non_epc_score:.3f} → {non_epc_class}")
