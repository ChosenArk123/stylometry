import os
import sys
import numpy as np
from tqdm import tqdm
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

    # Word Frequencies
    word_counts = {word: 0 for word in MARKER_WORDS}
    total_words = len([t for t in tokens if not t.is_punct])

    for token in tokens:
        t_lower = token.text.lower()
        if t_lower in word_counts:
            word_counts[t_lower] += 1

    marker_features = []
    if total_words > 0:
        marker_features = [word_counts[w] / total_words for w in MARKER_WORDS]
    else:
        marker_features = [0] * len(MARKER_WORDS)

    # Base features (Indices 0-6)
    base_features = [
        s_stats['mean_sentence_length'],  # 0
        s_stats['std_sentence_length'],  # 1
        lexical_diversity(tokens),  # 2
        average_syntactic_depth(sentences),  # 3
        ratios['function_ratio'],  # 4
        ratios['content_ratio'],  # 5
        flesch_kincaid_grade(tokens, sentences)  # 6
    ]

    return base_features + marker_features


def train_and_predict(training_dir, unknown_file):
    X = []
    y = []

    # 1. Count files for progress bar
    total_files = 0
    for author_name in os.listdir(training_dir):
        author_path = os.path.join(training_dir, author_name)
        if os.path.isdir(author_path):
            total_files += len([f for f in os.listdir(author_path) if f.endswith(".txt")])

    print(f"Training model on {total_files} samples...")
    sys.stdout.flush()

    # 2. Load Data
    with tqdm(total=total_files, desc="Processing Library", unit="file", file=sys.stdout, mininterval=0.1) as pbar:
        for author_name in os.listdir(training_dir):
            author_path = os.path.join(training_dir, author_name)
            if not os.path.isdir(author_path): continue

            for filename in os.listdir(author_path):
                if not filename.endswith(".txt"): continue

                filepath = os.path.join(author_path, filename)
                try:
                    text = load_text(filepath)
                    features = extract_features(text)
                    if features:
                        X.append(features)
                        y.append(author_name)
                except Exception:
                    pass

                pbar.update(1)
                sys.stdout.flush()

    if not X:
        print("Error: No valid training data found.")
        sys.stdout.flush()
        return

    # 3. Calculate Class Averages (Profiling)
    X_matrix = np.array(X)
    y_vector = np.array(y)
    unique_classes = np.unique(y_vector)
    class_profiles = {}

    for c in unique_classes:
        # Get all rows belonging to this class and average them
        class_rows = X_matrix[y_vector == c]
        class_profiles[c] = np.mean(class_rows, axis=0)

    # 4. Train
    clf = make_pipeline(StandardScaler(), SVC(probability=True))
    clf.fit(X, y)

    # 5. Predict
    print("\nAnalyzing unknown text...")
    sys.stdout.flush()
    unknown_text = load_text(unknown_file)
    unknown_features = extract_features(unknown_text)

    if not unknown_features:
        print("Error: Unknown file is unreadable.")
        sys.stdout.flush()
        return

    prediction = clf.predict([unknown_features])[0]
    probabilities = clf.predict_proba([unknown_features])[0]

    # --- RENAME LOGIC (Academic -> Scientific) ---
    display_prediction = "Scientific" if prediction == "Academic" else prediction

    print("\n=======================================================")
    print("  AI CLASSIFICATION RESULT")
    print("=======================================================")
    print(f"PREDICTION: The author is likely '{display_prediction}'")

    print("\nConfidence Breakdown:")
    for author, prob in zip(clf.classes_, probabilities):
        # Rename here too
        d_name = "Scientific" if author == "Academic" else author

        bar_len = int(prob * 20)
        bar = "█" * bar_len + "-" * (20 - bar_len)
        print(f"  {d_name.ljust(15)}: [{bar}] {prob * 100:.1f}%")

    # --- LOGIC EXPLAINED SECTION ---
    print("\n-------------------------------------------------------")
    print(f"Logic Explained: Why {display_prediction}?")
    print(f"Your text's structural fingerprint closely aligns with the")
    print(f"average profile of the '{display_prediction}' category:")

    # We compare the user's vector to the category's average vector
    # Indices: 2=Diversity, 3=Depth, 6=Grade
    metrics_map = [
        (6, "Reading Grade"),
        (3, "Syntactic Depth"),
        (2, "Lexical Diversity")
    ]

    cat_avg = class_profiles[prediction]  # Use original label to fetch profile

    for idx, name in metrics_map:
        user_val = unknown_features[idx]
        avg_val = cat_avg[idx]

        # Formatting
        diff = user_val - avg_val
        direction = "+" if diff > 0 else ""

        print(f"  • {name.ljust(18)}: {user_val:.2f}  (Category Avg: {avg_val:.2f})")

    print("\n(The AI chose the category where these numbers matched best.)")
    sys.stdout.flush()


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python -m analysis.classify <training_folder> <unknown_file>")
    else:
        train_and_predict(sys.argv[1], sys.argv[2])