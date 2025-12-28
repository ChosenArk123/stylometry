import spacy
from pathlib import Path
import pypdf
import docx
import re  # Added for noise filtering

nlp = spacy.load("en_core_web_sm")
nlp.max_length = 5000000


def load_text(path: str) -> str:
    path_obj = Path(path)
    suffix = path_obj.suffix.lower()

    if suffix == '.pdf':
        text = ""
        try:
            reader = pypdf.PdfReader(path)
            for page in reader.pages:
                extracted = page.extract_text()
                if extracted:
                    # FIX: PDFs often miss spaces at line breaks.
                    # We ensure a space exists to prevent word-merging.
                    text += extracted + " \n"
            return text
        except Exception as e:
            raise ValueError(f"Could not read PDF: {e}")
    # ... (keep docx and txt logic as is)
    return path_obj.read_text(encoding="utf-8")


def preprocess(text: str):
    # 1. CLEANING: Remove library-style noise (ISBNs, long numbers, page markers)
    # This prevents barcodes from being counted as 20-syllable words.
    text = re.sub(r'\d{5,}', '', text)  # Remove strings of 5+ digits
    text = re.sub(r'--- PAGE \d+ ---', '', text)  # Remove the page markers

    doc = nlp(text)

    # 2. SENTENCE FILTERING:
    # We ignore very short fragments (< 5 words) and extremely long
    # "Mega-Sentences" (> 150 words) which are likely extraction errors.
    sentences = [
        sent for sent in doc.sents
        if 5 < len(sent) < 150
    ]

    # 3. TOKEN FILTERING: Keep only actual words
    tokens = [
        token for token in doc
        if not token.is_space and not token.is_punct and token.is_alpha
    ]

    return doc, sentences, tokens