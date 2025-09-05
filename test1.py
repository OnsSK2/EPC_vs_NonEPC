from PIL import Image, ImageEnhance, ImageFilter
import pytesseract
from difflib import SequenceMatcher
from fuzzywuzzy import fuzz
import unicodedata

# Chemin vers l'exécutable Tesseract (ajuste selon ton installation)
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Mots-clés EPC avec pondération
EPC_KEYWORDS = {
    "Primärenergie": 3,
    "CO2": 3,
    "Heizbedarf": 3,
    "Energie": 2,
    "Passivhaus": 2,
    "Wohngebäude": 1,
    "Referenzgebäude": 1,
    "Gebäudehülle": 2,
    "Fenster": 1,
    "Dämmung": 2
}

# --- OCR avec prétraitement ---
def extract_text_from_image(image_path, lang='deu'):
    img = Image.open(image_path).convert('L')  # grayscale
    img = img.filter(ImageFilter.MedianFilter())  # réduit le bruit
    enhancer = ImageEnhance.Contrast(img)
    img = enhancer.enhance(2)  # augmente le contraste
    img = img.resize((img.width*2, img.height*2))  # agrandir pour OCR
    text = pytesseract.image_to_string(img, lang=lang)
    return text

# --- Similarité de texte ---
def compute_similarity(text1, text2):
    return SequenceMatcher(None, text1, text2).ratio()

# --- Normalisation du score ---
def normalize_score(score):
    return max(0.0, min(1.0, score))

# --- Score basé sur mots-clés avec fuzzy matching ---
def keyword_ratio(text):
    text_norm = unicodedata.normalize('NFKD', text.lower())
    total_weight = sum(EPC_KEYWORDS.values())
    found_weight = 0
    critical_keywords = ["Primärenergie", "CO2", "Heizbedarf"]
    critical_found = 0

    for kw, weight in EPC_KEYWORDS.items():
        kw_norm = unicodedata.normalize('NFKD', kw.lower())
        if fuzz.partial_ratio(kw_norm, text_norm) > 80:
            found_weight += weight
            if kw in critical_keywords:
                critical_found += 1

    ratio = found_weight / total_weight if total_weight > 0 else 0.0

    # Bonus si plusieurs mots-clés critiques sont trouvés
    if critical_found >= 2:
        ratio = min(1.0, ratio + 0.2)

    return ratio

# --- Classification selon le score ---
def classify_score(score, epc_threshold=0.88, non_epc_threshold=0.6):
    if score >= epc_threshold:
        return "EPC"
    elif score <= non_epc_threshold:
        return "Non-EPC"
    else:
        return "Indeterminate"

# --- Score final combinant similarité et mots-clés ---
def score_document(reference_image, candidate_image, alpha=0.25):
    ref_text = extract_text_from_image(reference_image)
    cand_text = extract_text_from_image(candidate_image)
    
    sim_text = compute_similarity(ref_text, cand_text)
    key_ratio = keyword_ratio(cand_text)
    
    final_score = alpha * sim_text + (1 - alpha) * key_ratio
    normalized_score = normalize_score(final_score)
    classification = classify_score(normalized_score)
    
    return normalized_score, classification

# --- Exemple d'utilisation ---
if __name__ == "__main__":
    reference_img = "reference.png"
    epc_candidate_img = "match.png"
    non_epc_candidate_img = "not_epc1.png"

    epc_score, epc_class = score_document(reference_img, epc_candidate_img)
    non_epc_score, non_epc_class = score_document(reference_img, non_epc_candidate_img)

    print(f"EPC match score: {epc_score:.3f} → {epc_class}")
    print(f"Non-EPC match score: {non_epc_score:.3f} → {non_epc_class}")
