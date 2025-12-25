import streamlit as st
import os
import tempfile
from pathlib import Path

# Import your modules
from analysis.preprocessing import load_text, preprocess
from analysis.metrics import (
    sentence_stats, lexical_diversity, average_syntactic_depth,
    flesch_kincaid_grade, get_stylometric_vector, cosine_similarity,
    sentence_lengths, get_top_ngrams, jaccard_similarity, function_content_ratio
)
from analysis.visualize import plot_comparison
# Note: We import matplotlib directly to pass figures to Streamlit
import matplotlib.pyplot as plt

st.set_page_config(page_title="Stylometry AI", layout="wide")

st.title("🕵️ Stylometry AI: Forensic Authorship Analysis")
st.markdown("""
This tool uses **Natural Language Processing (NLP)** and **Machine Learning** to analyze writing styles.
It can detect hidden habits, measure linguistic complexity, and predict authorship.
""")

# --- SIDEBAR ---
st.sidebar.header("Configuration")
mode = st.sidebar.radio("Select Mode", ["Single Analysis", "Comparative Analysis", "Deep Learning Classifier"])


# --- HELPER FUNCTIONS ---
def analyze_wrapper(file_obj):
    # Save uploaded file to temp so our existing 'load_text' can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_obj.name).suffix) as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name

    text = load_text(tmp_path)
    os.unlink(tmp_path)  # Clean up

    doc, sentences, tokens = preprocess(text)
    s_stats = sentence_stats(sentences)
    ratios = function_content_ratio(tokens)

    return {
        "filename": file_obj.name,
        "text": text,
        "doc": doc,
        "sentences": sentences,
        "tokens": tokens,
        "stats": {
            "mean_len": s_stats['mean_sentence_length'],
            "grade": flesch_kincaid_grade(tokens, sentences),
            "diversity": lexical_diversity(tokens),
            "depth": average_syntactic_depth(sentences),
            "func_ratio": ratios['function_ratio'],
            "content_ratio": ratios['content_ratio'],
            "raw_lens": sentence_lengths(sentences),
            "bigrams": get_top_ngrams(tokens, n=2, limit=5),
            "all_bigrams_set": set(
                [f"{x[0]} {x[1]}" for x in zip([t.text for t in tokens], [t.text for t in tokens][1:])])
            # Simplified for app
        }
    }


# --- MODE 1: SINGLE ANALYSIS ---
if mode == "Single Analysis":
    uploaded_file = st.file_uploader("Upload a Text/PDF/Docx", type=['txt', 'pdf', 'docx'])

    if uploaded_file:
        with st.spinner("Analyzing linguistic structure..."):
            data = analyze_wrapper(uploaded_file)

        # Display Dashboard
        st.header(f"Report: {data['filename']}")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Grade Level", f"{data['stats']['grade']:.1f}")
        col2.metric("Avg Sent Length", f"{data['stats']['mean_len']:.1f}")
        col3.metric("Lexical Diversity", f"{data['stats']['diversity']:.3f}")
        col4.metric("Syntactic Depth", f"{data['stats']['depth']:.2f}")

        st.subheader("Top Phrasing Habits")
        st.write(", ".join([f"**{p[0]}** ({p[1]})" for p in data['stats']['bigrams']]))

# --- MODE 2: COMPARATIVE ANALYSIS ---
elif mode == "Comparative Analysis":
    col1, col2 = st.columns(2)
    file_a = col1.file_uploader("Reference Text (A)", type=['txt', 'pdf', 'docx'])
    file_b = col2.file_uploader("Target Text (B)", type=['txt', 'pdf', 'docx'])

    if file_a and file_b:
        if st.button("Compare Styles"):
            with st.spinner("Calculating Vector Distance..."):
                data_a = analyze_wrapper(file_a)
                data_b = analyze_wrapper(file_b)

                # Manual plot logic for Streamlit
                fig, axes = plt.subplots(1, 2, figsize=(15, 5))

                # Histograms
                axes[0].hist(data_a['stats']['raw_lens'], alpha=0.5, label="File A", density=True, bins=15)
                axes[0].hist(data_b['stats']['raw_lens'], alpha=0.5, label="File B", density=True, bins=15)
                axes[0].set_title("Sentence Rhythm")
                axes[0].legend()

                # Bar Chart
                metrics = ['Grade', 'Diversity', 'Depth']
                vals_a = [data_a['stats']['grade'], data_a['stats']['diversity'] * 10, data_a['stats']['depth']]
                vals_b = [data_b['stats']['grade'], data_b['stats']['diversity'] * 10, data_b['stats']['depth']]
                x = [0, 1, 2]
                width = 0.35
                axes[1].bar([i - width / 2 for i in x], vals_a, width, label="File A")
                axes[1].bar([i + width / 2 for i in x], vals_b, width, label="File B")
                axes[1].set_xticks(x)
                axes[1].set_xticklabels(metrics)
                axes[1].set_title("Metric Comparison")
                axes[1].legend()

                st.pyplot(fig)

                st.info("Note: Diversity is scaled x10 for visibility.")

# --- MODE 3: DEEP LEARNING CLASSIFIER ---
elif mode == "Deep Learning Classifier":
    st.warning("⚠️ This mode requires a 'data/training' folder with labeled subfolders.")

    unknown = st.file_uploader("Upload Mystery Text", type=['txt', 'pdf', 'docx'])

    if unknown and st.button("Identify Author (BERT)"):
        with st.spinner("Loading Neural Network & Analyzing..."):
            # We call your CLI script's logic here, but adapted slightly
            from analysis.classify import train_and_predict

            # Save upload to temp
            with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                tmp.write(unknown.getvalue())
                unknown_path = tmp.name

            # Redirect stdout to capture the print statements from classify.py
            # (A bit of a hack, but works nicely to reuse your existing code)
            import io
            from contextlib import redirect_stdout

            f = io.StringIO()
            with redirect_stdout(f):
                train_and_predict("data/training", unknown_path)

            output = f.getvalue()
            st.text_area("Analysis Log", output, height=300)
            os.unlink(unknown_path)