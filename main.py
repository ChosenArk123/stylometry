import argparse
import sys
import tkinter as tk
from tkinter import filedialog
from pathlib import Path
from analysis.preprocessing import load_text, preprocess
from analysis.metrics import (
    sentence_stats, lexical_diversity, average_syntactic_depth,
    function_content_ratio, flesch_kincaid_grade,
    get_stylometric_vector, cosine_similarity, sentence_lengths,
    get_top_ngrams, get_ngrams, jaccard_similarity
)
from analysis.visualize import plot_comparison


def pick_file(title="Select File"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title)
    root.destroy()
    return file_path


def analyze_text(path):
    print(f"Processing: {Path(path).name}...")
    text = load_text(path)
    doc, sentences, tokens = preprocess(text)

    sent_stats = sentence_stats(sentences)
    ratios = function_content_ratio(tokens)

    results = {
        "filename": Path(path).name,
        "raw_sentence_lengths": sentence_lengths(sentences),
        "mean_sentence_length": sent_stats["mean_sentence_length"],
        "lexical_diversity": lexical_diversity(tokens),
        "average_syntactic_depth": average_syntactic_depth(sentences),
        "function_ratio": ratios["function_ratio"],
        "content_ratio": ratios["content_ratio"],
        "readability_grade": flesch_kincaid_grade(tokens, sentences),
        "all_bigrams": get_ngrams(tokens, n=2),
        "top_bigrams": get_top_ngrams(tokens, n=2, limit=5)
    }
    return results


def print_report(stats):
    print(f"\n=======================================================")
    print(f"  REPORT: {stats['filename']}")
    print(f"=======================================================")

    print(f"1. COMPLEXITY METRICS")
    print(f"   Readability Grade: {stats['readability_grade']:.1f}")
    print(f"     -> (Years of education needed to understand this text)")
    print(f"   Avg Sentence Length: {stats['mean_sentence_length']:.1f} words")
    print(f"     -> (Longer = usually more formal/academic)")
    print(f"   Avg Syntactic Depth: {stats['average_syntactic_depth']:.2f}")
    print(f"     -> (2-3 is Simple. 6+ is Highly Complex/Nested)")

    print(f"\n2. VOCABULARY METRICS")
    print(f"   Lexical Diversity: {stats['lexical_diversity']:.3f}")
    print(f"     -> (0.4 = Repetitive. 0.7 = Very rich/varied vocabulary)")
    print(f"   Function Word Ratio: {stats['function_ratio']:.2f}")
    print(f"     -> (Higher = More 'glue' words like 'the/and/of'. Lower = Information dense)")

    print(f"\n3. HABIT METRICS (Top 2-word phrases)")
    for phrase, count in stats['top_bigrams']:
        print(f"   - '{phrase}': {count} times")


def main():
    parser = argparse.ArgumentParser(description="Stylometry: Analyze and compare writing styles.")
    parser.add_argument("target", help="Path to the primary text file to analyze", nargs='?')
    parser.add_argument("--compare", help="Path to a second file to compare against", default=None)
    parser.add_argument("--visualize", help="Show graphs of the comparison", action="store_true")

    args = parser.parse_args()

    # 1. Handle Input
    target_path = args.target
    compare_path = args.compare

    if not target_path:
        print(">> No file provided. Opening file picker...")
        target_path = pick_file("Select Target Text (PDF, Docx, or Txt)")
        if not target_path:
            print("No file selected. Exiting.")
            sys.exit(0)

        print(f"Selected: {target_path}")
        mode = input("Do you want to compare this against another file? (y/n): ").lower()
        if mode.startswith('y'):
            compare_path = pick_file("Select Comparison File")
            if compare_path:
                args.visualize = True

    # 2. Analyze Target
    try:
        target_stats = analyze_text(target_path)
        print_report(target_stats)
    except Exception as e:
        print(f"Error processing target: {e}")
        sys.exit(1)

    # 3. Analyze Comparison
    if compare_path:
        try:
            comp_stats = analyze_text(compare_path)
            print_report(comp_stats)

            vec_a = get_stylometric_vector(target_stats)
            vec_b = get_stylometric_vector(comp_stats)

            similarity = cosine_similarity(vec_a, vec_b)
            jaccard = jaccard_similarity(target_stats['all_bigrams'], comp_stats['all_bigrams'])

            print(f"\n=======================================================")
            print(f"  COMPARATIVE VERDICT")
            print(f"=======================================================")
            print(f"Structural Similarity (Cosine): {similarity:.4f}")
            print(f"  -> (1.00 = Identical 'skeleton' of writing. <0.80 = Very different styles)")
            print(f"Phrasing Overlap (Jaccard):     {jaccard:.4f}")
            print(f"  -> (Percentage of 2-word phrases shared between texts)")

            print("\nAI CONCLUSION:")
            if similarity > 0.9 and jaccard > 0.1:
                print(">> High likelihood of same author.")
                print("   (Both the structure AND specific phrases match.)")
            elif similarity > 0.9:
                print(">> Similar Genre/Education Level.")
                print("   (Structure is similar, but they use different specific words.)")
            elif jaccard > 0.15:
                print(">> Suspected Copy-Paste / Mimicry.")
                print("   (Writing style is different, but they share A LOT of specific phrases.)")
            else:
                print(">> Distinct Authors.")
                print("   (These texts have little in common structurally or linguistically.)")

            if args.visualize:
                print("\nGenerating visualization dashboard...")
                plot_comparison(target_stats, comp_stats, target_stats['filename'], comp_stats['filename'])

        except Exception as e:
            print(f"Error processing comparison file: {e}")
            sys.exit(1)


if __name__ == "__main__":
    main()