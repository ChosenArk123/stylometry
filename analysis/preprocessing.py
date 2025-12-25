import spacy
from pathlib import Path
import pypdf
import docx

nlp = spacy.load("en_core_web_sm")


def load_text(path: str) -> str:
    path_obj = Path(path)

    if not path_obj.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path_obj.suffix.lower()

    if suffix == '.pdf':
        print(f"  > Detected PDF: Extracting text...")
        text = ""
        try:
            reader = pypdf.PdfReader(path)
            for page in reader.pages:
                text += page.extract_text() + "\n"
            return text
        except Exception as e:
            raise ValueError(f"Could not read PDF: {e}")

    elif suffix == '.docx':
        print(f"  > Detected Word Doc: Extracting text...")
        try:
            doc = docx.Document(path)
            return "\n".join([para.text for para in doc.paragraphs])
        except Exception as e:
            raise ValueError(f"Could not read DOCX: {e}")

    else:
        # Default to plain text (handles .txt, .md, .py, etc.)
        try:
            return path_obj.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            # Fallback for weird encodings
            return path_obj.read_text(encoding="latin-1")


def preprocess(text: str):
    doc = nlp(text)

    sentences = [
        sent for sent in doc.sents
        if len(sent) > 3
    ]

    tokens = [
        token for token in doc
        if not token.is_space
    ]

    return doc, sentences, tokens