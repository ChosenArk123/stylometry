import os
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from analysis.preprocessing import load_text, preprocess
from analysis.metrics import (
    sentence_stats, lexical_diversity, average_syntactic_depth,
    function_content_ratio, flesch_kincaid_grade
)

MARKER_WORDS = [
    "the", "and", "of", "a", "in", "to", "is", "that", "it", "with",
    "as", "for", "was", "on", "but", "are", "be", "not", "by", "at",
    "this", "which", "from", "or", "have", "an", "they", "their", "we",
    "however", "therefore", "thus", "although", "because"
]


def extract_features(text):
    doc, sentences, tokens = preprocess(text)
    if not sentences: return None

    s_stats = sentence_stats(sentences)
    ratios = function_content_ratio(tokens)

    # --- NEW: Calculate Word Frequencies ---
    # We count how often each marker word appears relative to total length
    word_counts = {word: 0 for word in MARKER_WORDS}
    total_words = len([t for t in tokens if not t.is_punct])

    for token in tokens:
        t_lower = token.text.lower()
        if t_lower in word_counts:
            word_counts[t_lower] += 1

    # Normalize (count / total_words)
    # If total_words is 0, avoid division by zero
    marker_features = []
    if total_words > 0:
        marker_features = [word_counts[w] / total_words for w in MARKER_WORDS]
    else:
        marker_features = [0] * len(MARKER_WORDS)
    # ---------------------------------------

    # Combine structural stats with word stats
    base_features = [
        s_stats['mean_sentence_length'],
        s_stats['std_sentence_length'],
        lexical_diversity(tokens),
        average_syntactic_depth(sentences),
        ratios['function_ratio'],
        ratios['content_ratio'],
        flesch_kincaid_grade(tokens, sentences)
    ]

    return base_features + marker_features

def train_and_predict(training_dir, unknown_file):
    """
    1. Reads all files in 'training_dir'.
    2. Assumes folder names are the author labels (e.g. data/authors/hemingway/*.txt).
    3. Trains a Support Vector Machine (SVM).
    4. Predicts the author of 'unknown_file'.
    """
    X = []  # Features
    y = []  # Labels (Authors)

    print("Training model...")

    # Walk through the directory
    # Expected structure: data/authors/AUTHOR_NAME/file.txt
    for author_name in os.listdir(training_dir):
        author_path = os.path.join(training_dir, author_name)

        if not os.path.isdir(author_path): continue

        print(f"  Loading author: {author_name}")
        for filename in os.listdir(author_path):
            if not filename.endswith(".txt"): continue

            filepath = os.path.join(author_path, filename)
            text = load_text(filepath)
            features = extract_features(text)

            if features:
                X.append(features)
                y.append(author_name)

    if not X:
        print("Error: No valid training data found.")
        return

    # Create a Classifier Pipeline
    # StandardScaler normalizes data (so 'sentence length 20' doesn't outweigh 'ratio 0.5')
    clf = make_pipeline(StandardScaler(), SVC(probability=True))
    clf.fit(X, y)

    print("\nAnalyzing unknown text...")
    unknown_text = load_text(unknown_file)
    unknown_features = extract_features(unknown_text)

    if not unknown_features:
        print("Error: Unknown file is empty.")
        return

    # Predict
    prediction = clf.predict([unknown_features])[0]
    probabilities = clf.predict_proba([unknown_features])[0]

    print(f"\n=======================================================")
    print(f"  AI CLASSIFICATION RESULT")
    print(f"=======================================================")
    print(f"PREDICTION: The author is likely '{prediction}'")

    print("\nConfidence Breakdown:")
    print("(How sure is the model about each possibility?)")
    for author, prob in zip(clf.classes_, probabilities):
        bar_len = int(prob * 20)
        bar = "█" * bar_len + "-" * (20 - bar_len)
        print(f"  {author.ljust(15)}: [{bar}] {prob * 100:.1f}%")


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 3:
        print("Usage: python -m analysis.classify <training_folder> <unknown_file>")
        print("Example: python -m analysis.classify data/authors data/unknown.txt")
    else:
        train_and_predict(sys.argv[1], sys.argv[2])