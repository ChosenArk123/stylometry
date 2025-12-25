import torch
import numpy as np
from transformers import DistilBertTokenizer, DistilBertModel
from analysis.preprocessing import load_text

# Global variables to cache the model (so we don't reload it 50 times)
_tokenizer = None
_model = None


def load_bert():
    """Loads the DistilBERT model. Only runs once."""
    global _tokenizer, _model
    if _model is None:
        print("  > Loading Neural Network (DistilBERT)...")
        _tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        _model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        _model.eval()  # Set to evaluation mode (faster, no training)


def get_bert_embedding(text):
    """
    Feeds text into the Neural Network and returns a 768-dimensional vector
    representing the 'meaning' and 'style' of the text.
    """
    load_bert()

    # 1. Tokenize (Turn words into ID numbers the AI understands)
    # We truncate to 512 tokens because BERT has a limit.
    inputs = _tokenizer(text, return_tensors="pt", max_length=512, truncation=True, padding=True)

    # 2. Feed to Model
    with torch.no_grad():  # Disable gradient calculation (saves RAM)
        outputs = _model(**inputs)

    # 3. Get the "Hidden State" (The brain's representation)
    # We take the mean (average) of all token vectors to get one vector for the whole text.
    # shape: [batch_size, sequence_length, hidden_size] -> [hidden_size]
    last_hidden_states = outputs.last_hidden_state
    mean_embedding = last_hidden_states.mean(dim=1).squeeze().numpy()

    return mean_embedding


if __name__ == "__main__":
    # Test it out
    vec = get_bert_embedding("The quick brown fox jumps over the lazy dog.")
    print(f"Generated Vector Shape: {vec.shape}")
    print(f"First 10 features: {vec[:10]}")