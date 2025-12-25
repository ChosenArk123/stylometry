import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from analysis.preprocessing import load_text, preprocess
from analysis.metrics import (
    sentence_stats, lexical_diversity, average_syntactic_depth,
    function_content_ratio, flesch_kincaid_grade
)


# --- COPY YOUR EXTRACT_FEATURES FUNCTION HERE ---
# (Or import it if you refactor classify.py to allow imports)
def extract_features(text):
    doc, sentences, tokens = preprocess(text)
    if not sentences: return None
    s_stats = sentence_stats(sentences)
    ratios = function_content_ratio(tokens)
    return [
        s_stats['mean_sentence_length'],
        s_stats['std_sentence_length'],
        lexical_diversity(tokens),
        average_syntactic_depth(sentences),
        ratios['function_ratio'],
        ratios['content_ratio'],
        flesch_kincaid_grade(tokens, sentences)
    ]


# -----------------------------------------------

def map_authors(training_dir):
    print("Extracting features for visualization...")

    X = []
    labels = []
    filenames = []

    # 1. Load Data
    for author in os.listdir(training_dir):
        author_path = os.path.join(training_dir, author)
        if not os.path.isdir(author_path): continue

        print(f"  > Processing {author}...")
        for fname in os.listdir(author_path):
            if not fname.endswith(".txt"): continue

            text = load_text(os.path.join(author_path, fname))
            feats = extract_features(text)

            if feats:
                X.append(feats)
                labels.append(author)
                filenames.append(fname)

    if not X:
        print("No data found!")
        return

    # 2. Normalize Data (StandardScaler)
    # PCA is very sensitive to scale. We must scale the data first.
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 3. Apply PCA (7D -> 2D)
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

    # 4. Plot
    plt.figure(figsize=(10, 8))

    # Get unique authors to assign colors
    unique_authors = list(set(labels))
    colors = plt.cm.get_cmap('tab10', len(unique_authors))

    for i, author in enumerate(unique_authors):
        # Find indices where label matches current author
        idxs = [j for j, label in enumerate(labels) if label == author]

        # Plot those points
        plt.scatter(
            X_pca[idxs, 0],
            X_pca[idxs, 1],
            label=author,
            s=50,  # size
            alpha=0.7  # transparency
        )

    plt.title('Stylometric Map (PCA)', fontsize=16)
    plt.xlabel(f'Principal Component 1 ({pca.explained_variance_ratio_[0] * 100:.1f}% Variance)', fontsize=12)
    plt.ylabel(f'Principal Component 2 ({pca.explained_variance_ratio_[1] * 100:.1f}% Variance)', fontsize=12)
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)

    print("Displaying map...")
    plt.show()


if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python -m analysis.map_style <training_dir>")
    else:
        map_authors(sys.argv[1])