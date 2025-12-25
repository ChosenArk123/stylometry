import matplotlib.pyplot as plt
import numpy as np


def plot_comparison(stats_a, stats_b, name_a="Text A", name_b="Text B"):
    """
    Generates a dashboard of 3 charts comparing the two texts,
    with added explanatory text for the user.
    """
    # Increase figure height to make room for the explanation at the bottom
    fig, axes = plt.subplots(1, 3, figsize=(18, 7))
    fig.suptitle(f'Stylometric Comparison: {name_a} vs {name_b}', fontsize=16, weight='bold')

    # Chart 1: Sentence Length Distribution
    lens_a = stats_a['raw_sentence_lengths']
    lens_b = stats_b['raw_sentence_lengths']

    axes[0].hist(lens_a, alpha=0.5, label=name_a, bins=20, density=True, color='blue')
    axes[0].hist(lens_b, alpha=0.5, label=name_b, bins=20, density=True, color='orange')
    axes[0].set_title('Sentence Length Rhythm')
    axes[0].set_xlabel('Words per Sentence')
    axes[0].set_ylabel('Frequency')
    axes[0].legend()

    # Chart 2: Key Metrics
    metrics = ['Lexical Div', 'Avg Depth', 'Func Ratio', 'Readability']
    values_a = [stats_a['lexical_diversity'], stats_a['average_syntactic_depth'], stats_a['function_ratio'],
                stats_a['readability_grade']]
    values_b = [stats_b['lexical_diversity'], stats_b['average_syntactic_depth'], stats_b['function_ratio'],
                stats_b['readability_grade']]

    x = np.arange(len(metrics))
    width = 0.35

    axes[1].bar(x - width / 2, values_a, width, label=name_a, color='blue')
    axes[1].bar(x + width / 2, values_b, width, label=name_b, color='orange')
    axes[1].set_title('Stylometric Markers')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(metrics)
    axes[1].legend()

    # Chart 3: Function vs Content
    labels = [name_a, name_b]
    content_ratios = [stats_a['content_ratio'], stats_b['content_ratio']]
    function_ratios = [stats_a['function_ratio'], stats_b['function_ratio']]

    axes[2].bar(labels, content_ratios, label='Content Words (Nouns/Verbs)', color='green')
    axes[2].bar(labels, function_ratios, bottom=content_ratios, label='Function Words (The/And/Of)', color='gray')
    axes[2].set_title('Vocabulary Composition')
    axes[2].legend()

    # --- ADDED EXPLANATION TEXT ---
    explanation = (
        "HOW TO READ THIS GRAPH:\n"
        "1. Rhythm: Do the graphs overlap? If yes, they use similar sentence structures.\n"
        "2. Lexical Div: Higher bar = Richer vocabulary (less repetition).\n"
        "3. Avg Depth: Higher bar = More complex grammar (nested clauses).\n"
        "4. Func Ratio: High gray bar = 'Glue' words (conversational). High green bar = 'Content' words (academic)."
    )
    plt.figtext(0.5, 0.02, explanation, ha="center", fontsize=10,
                bbox={"facecolor": "orange", "alpha": 0.2, "pad": 5})
    # ------------------------------

    plt.tight_layout(rect=[0, 0.1, 1, 1])  # Adjust layout to make room for text
    plt.show()