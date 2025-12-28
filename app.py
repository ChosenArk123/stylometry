import streamlit as st
import os
import tempfile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# --- PROJECT IMPORTS ---
from analysis.preprocessing import load_text, preprocess
from analysis.metrics import (
    sentence_stats, lexical_diversity, average_syntactic_depth,
    flesch_kincaid_grade, get_stylometric_vector, cosine_similarity,
    sentence_lengths, get_top_ngrams, jaccard_similarity, function_content_ratio
)
from analysis.classify import train_and_predict

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stylometry Ultra", layout="wide", page_icon="🕵️")

# --- STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 20px;
        border-radius: 10px;
        border-left: 6px solid #4e73df;
        margin-bottom: 20px;
        box-shadow: 2px 2px 10px rgba(0,0,0,0.05);
    }
    .stat-name { font-size: 22px; font-weight: bold; color: #2e59d9; }
    .explanation { font-size: 14px; color: #444; margin-top: 8px; line-height: 1.5; }
    .data-point { font-size: 28px; font-weight: bold; color: #000; }
    .range-breakdown { 
        margin-top: 10px; 
        padding: 10px; 
        background-color: #ffffff; 
        border-radius: 5px; 
        border: 1px solid #e3e6f0;
        font-size: 13px;
        color: #333333;
    }
    .verdict-box {
        padding: 25px;
        border-radius: 12px;
        background-color: #f1f3f5;
        border: 1px solid #dee2e6;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Stylometry Analysis Suite")
st.markdown("### Forensic Linguistic Profiling & Deep Authorship Analysis")


# --- DATA SCALES & EXPLANATIONS ---
def get_detailed_explanation(name, value):
    """Provides a detailed name, value, and granular range-based explanation."""

    if name == "Reading Grade (Flesch-Kincaid)":
        if value <= 6:
            level = "Elementary (Basic)"
        elif value <= 10:
            level = "Middle/High School (Standard)"
        elif value <= 14:
            level = "College (Professional)"
        else:
            level = "Post-Graduate (Academic/Technical)"

        return {
            "val": f"Grade {value:.1f} — {level}",
            "desc": "This metric estimates the U.S. school grade level required to understand the text.",
            "ranges": [
                "0–6: Simple, direct language. Children's literature or basic instructions.",
                "7–10: Journalistic/Essayist. Accessible standard prose (e.g. Orwell, Montaigne).",  # <--- UPDATE THIS
                "11–14: Novelist/Academic. The sweet spot for complex fiction or university research.",
                "15+: Legal/Specialist. Contracts, medical journals, and court opinions."
            ]
        }

    if name == "Lexical Diversity (TTR)":
        if value < 0.40:
            level = "Repetitive (Legal/Technical)"
        elif value < 0.60:
            level = "Balanced (Standard)"
        else:
            level = "Rich (Novelist)"

        return {
            "val": f"{value:.3f} — {level}",
            "desc": "Type-Token Ratio (TTR) measures vocabulary richness.",
            "ranges": [
                "< 0.40: Legal/Technical. Repetitive terms required for precision.",
                "0.40–0.60: Essayist/Standard. Balances clarity with varied vocabulary.",  # <--- UPDATE THIS
                "> 0.60: Novelist/Poetic. Fiction writers deliberately avoid repetition."
            ]
        }

    if name == "Syntactic Depth":
        if value <= 3.0:
            level = "Basic (Flat)"
        elif value <= 5.0:
            level = "Modern Standard (Integrated)"
        elif value <= 7.5:
            level = "Sophisticated (Academic)"
        else:
            level = "Extremely Dense (Legal)"

        return {
            "val": f"{value:.2f} — {level}",
            "desc": "Measures grammatical complexity via 'nested' clauses.",
            "ranges": [
                "1.0–3.0: Flat syntax. Direct Subject-Verb-Object sentences.",
                "3.1–5.0: Essayist/Narrative. Uses dependent clauses to argue points or show action.",
                # <--- UPDATE THIS
                "5.1–7.5: Academic. Heavy use of commas, semicolons, and qualifiers.",
                "> 7.5: Legal. Extreme complexity common in statutes."
            ]
        }

    if name == "Content Word Ratio":
        if value < 0.45:
            level = "Conversational (Glue-Heavy)"
        elif value <= 55:
            level = "Narrative (Balanced)"
        else:
            level = "Information-Dense (Hard)"

        return {
            "val": f"{value * 100:.1f}% — {level}",
            "desc": "The percentage of Nouns, Verbs, and Adjectives. The remainder is 'Glue' words (the, of, and, which).",
            "ranges": [
                "< 45%: Spoken tone. Relies heavily on pronouns and articles. Feels 'soft'.",
                "45–55%: Novelist/Narrative. The standard zone for storytelling.",
                "> 55%: Academic/Legal. Signals a focus on facts, definitions, and objects rather than flow."
            ]
        }

    if name == "Sentence Consistency (Std Dev)":
        if value < 5:
            level = "Monotone (Predictable)"
        elif value <= 15:
            level = "Dynamic (Novelist)"
        else:
            level = "Erratic (High Contrast)"

        return {
            "val": f"{value:.2f} — {level}",
            "desc": "Measures the variance in sentence length. High values mean the author 'mixes it up' for better pacing.",
            "ranges": [
                "< 5: Robotic. Almost every sentence is the same length. Common in technical lists.",
                "5–15: Novelist. The author varies length (dialogue vs. description) to keep the reader engaged.",
                "> 15: Stylistic extremes. Frequent use of 2-word punchy sentences alongside 60-word run-ons."
            ]
        }

    return {"val": str(value), "desc": "", "ranges": []}


# --- CACHED CORE LOGIC ---
@st.cache_data(show_spinner=False)
def perform_deep_analysis(text, filename):
    doc, sentences, tokens = preprocess(text)
    s_stats = sentence_stats(sentences)
    ratios = function_content_ratio(tokens)
    grade = flesch_kincaid_grade(tokens, sentences)
    diversity = lexical_diversity(tokens)
    depth = average_syntactic_depth(sentences)
    raw_lens = sentence_lengths(sentences)
    top_bigrams = get_top_ngrams(tokens, n=2, limit=10)
    all_bigrams = [f"{x[0]} {x[1]}" for x in zip([t.text for t in tokens], [t.text for t in tokens][1:])]

    vector = np.array([s_stats['mean_sentence_length'], diversity, depth, ratios['function_ratio'], grade])

    return {
        "filename": filename,
        "stats": {
            "Reading Grade (Flesch-Kincaid)": grade,
            "Lexical Diversity (TTR)": diversity,
            "Syntactic Depth": depth,
            "Content Word Ratio": ratios['content_ratio'],
            "Avg Sentence Length": s_stats['mean_sentence_length'],
            "Sentence Consistency (Std Dev)": s_stats['std_sentence_length'],
            "func_ratio": ratios['function_ratio'],
            "raw_lens": raw_lens
        },
        "top_bigrams": top_bigrams,
        "all_bigrams": all_bigrams,
        "vector": vector
    }


def load_and_cache(uploaded_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(uploaded_file.name).suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        text = load_text(tmp_path)
        return perform_deep_analysis(text, uploaded_file.name), text
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)


# --- UI LOGIC ---
mode = st.sidebar.radio("Select Analysis Type:", ["Single Text Analysis", "Comparative Analysis"])

if mode == "Single Text Analysis":
    uploaded = st.file_uploader("Upload Document (PDF, TXT, DOCX)", type=['txt', 'pdf', 'docx'])

    if uploaded:
        with st.spinner("Analyzing linguistic fingerprints..."):
            data, full_text = load_and_cache(uploaded)

        st.header(f"📄 Full Analysis Report: {data['filename']}")

        # Display Metric Cards
        cols = st.columns(2)
        idx = 0
        for name, value in data['stats'].items():
            if name in ["func_ratio", "raw_lens", "Avg Sentence Length"]: continue
            info = get_detailed_explanation(name, value)
            with cols[idx % 2]:
                st.markdown(f"""
                <div class="metric-card">
                    <div class="stat-name">{name}</div>
                    <div class="data-point">{info['val']}</div>
                    <div class="explanation"><b>Definition:</b> {info['desc']}</div>
                    <div class="range-breakdown"><b>Reference Scales:</b><br/>{"<br/>".join(info['ranges'])}</div>
                </div>
                """, unsafe_allow_html=True)
            idx += 1

        st.markdown("---")
        st.header("📊 Deep Dive: Visual Profiles")
        st.markdown("""
        Visual profiles provide a look at the subconscious habits of the writer. 
        While an author can choose their words, they rarely choose their 'Sentence Rhythm' or 'Vocabulary Architecture' consciously.
        """)

        g1, g2 = st.columns(2)

        with g1:
            st.subheader("1. Sentence Rhythm Distribution")
            st.markdown("""
            What this does: This graph plots the length of every sentence in the text. 
            * A high peak at one number: Means the author is very consistent but potentially monotone.
            * A wide, flat curve: Means the author varies their pacing constantly, mixing short 'punches' with long descriptions.
            """)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(data['stats']['raw_lens'], kde=True, bins=20, color="#4e73df", ax=ax1)
            ax1.set_xlabel("Words per Sentence")
            st.pyplot(fig1)

        with g2:
            st.subheader("2. Vocabulary Architecture")
            st.markdown("""
            What this does: This breaks down the 'building blocks' of the text.
            * Content Words: Nouns and verbs that carry meaning. Higher ratios signal academic or descriptive writing.
            * Function Words: Pronouns and articles that provide structure. Higher ratios signal a conversational or 'character-driven' style.
            """)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            ax2.pie([data['stats']['func_ratio'], data['stats']['Content Word Ratio']],
                    labels=['Function Words (Glue)', 'Content Words (Meat)'], autopct='%1.1f%%',
                    colors=['#f6c23e', '#1cc88a'])
            st.pyplot(fig2)

        st.markdown("---")
        st.subheader("3. Behavioral Fingerprint (Top Phrases)")
        st.markdown("""
        What this does: Also known as 'n-grams.' Every writer has 'filler' phrases or habitual word pairings they use subconsciously. 
        This chart exposes the specific 2-word combinations that define this author's signature style.
        """)
        if data['top_bigrams']:
            phrases, counts = zip(*data['top_bigrams'])
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            ax3.barh(np.arange(len(phrases)), counts, color='#36b9cc')
            ax3.set_yticks(np.arange(len(phrases)))
            ax3.set_yticklabels(phrases)
            ax3.invert_yaxis()
            st.pyplot(fig3)

        # Prediction Section
        st.markdown("---")
        st.subheader("🕵️ Authorship Prediction")
        if os.path.exists("data/training") and len(os.listdir("data/training")) > 0:
            if st.button("Predict Identity"):
                with tempfile.NamedTemporaryFile(delete=False, suffix=".txt") as tmp:
                    tmp.write(full_text.encode('utf-8'))
                    p_path = tmp.name
                try:
                    import io
                    from contextlib import redirect_stdout

                    f = io.StringIO()
                    with redirect_stdout(f):
                        train_and_predict("data/training", p_path)
                    st.text_area("AI Forensic Prediction", f.getvalue(), height=250)
                finally:
                    if os.path.exists(p_path): os.unlink(p_path)
        else:
            st.warning("⚠️ Training data required for identity prediction.")

elif mode == "Comparative Analysis":
    st.sidebar.info("Upload two documents to find stylistic overlap.")
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Reference Text A", type=['txt', 'pdf', 'docx'])
    f2 = c2.file_uploader("Candidate Text B", type=['txt', 'pdf', 'docx'])

    if f1 and f2:
        if st.button("🚀 Run Comparative Forensic Scan"):
            d1, _ = load_and_cache(f1)
            d2, _ = load_and_cache(f2)
            cos_sim = cosine_similarity(d1['vector'], d2['vector'])
            jac_sim = jaccard_similarity(d1['all_bigrams'], d2['all_bigrams'])

            st.markdown("<div class='verdict-box'>", unsafe_allow_html=True)
            st.subheader("🏛️ Comparative Verdict")
            v1, v2 = st.columns(2)
            v1.metric("Structural Match (Cosine)", f"{cos_sim:.4f}")
            v2.metric("Phrase Overlap (Jaccard)", f"{jac_sim:.4f}")

            st.write("Interpretation:")
            if cos_sim > 0.95:
                st.success(
                    "The structural 'skeletons' are nearly identical, suggesting the same author or a very successful imitation.")
            elif cos_sim > 0.85:
                st.info("Styles are similar, common in writers of the same genre or time period.")
            else:
                st.error("Styles are fundamentally distinct; authorship by the same person is unlikely.")

            if jac_sim > 0.20: st.warning(
                "High phrase overlap detected. This indicates shared linguistic habits or potential copy-pasting.")
            st.markdown("</div>", unsafe_allow_html=True)

            st.subheader("📊 Side-by-Side Visualization")
            st.write("1. Pacing Comparison (Sentence Rhythm)")
            fig_r, ax_r = plt.subplots(figsize=(10, 3))
            sns.histplot(d1['stats']['raw_lens'], color="#4e73df", label="Text A", kde=True, stat="density", alpha=0.3,
                         ax=ax_r)
            sns.histplot(d2['stats']['raw_lens'], color="#f6c23e", label="Text B", kde=True, stat="density", alpha=0.3,
                         ax=ax_r)
            ax_r.legend()
            st.pyplot(fig_r)

            vg1, vg2 = st.columns(2)
            with vg1:
                st.write("2. Complexity Delta")
                v_a = [d1['stats']['Reading Grade (Flesch-Kincaid)'], d1['stats']['Lexical Diversity (TTR)'] * 10,
                       d1['stats']['Syntactic Depth']]
                v_b = [d2['stats']['Reading Grade (Flesch-Kincaid)'], d2['stats']['Lexical Diversity (TTR)'] * 10,
                       d2['stats']['Syntactic Depth']]
                fig_b, ax_b = plt.subplots()
                x = np.arange(3)
                ax_b.bar(x - 0.175, v_a, 0.35, label='Text A', color='#4e73df')
                ax_b.bar(x + 0.175, v_b, 0.35, label='Text B', color='#f6c23e')
                ax_b.set_xticks(x)
                ax_b.set_xticklabels(['Grade', 'Diversity', 'Depth'])
                ax_b.legend()
                st.pyplot(fig_b)

            with vg2:
                st.write("3. Density Comparison")
                fig_s, ax_s = plt.subplots()
                ax_s.bar(['Text A', 'Text B'], [d1['stats']['Content Word Ratio'], d2['stats']['Content Word Ratio']],
                         label='Content', color='#1cc88a')
                ax_s.bar(['Text A', 'Text B'], [d1['stats']['func_ratio'], d2['stats']['func_ratio']],
                         bottom=[d1['stats']['Content Word Ratio'], d2['stats']['Content Word Ratio']],
                         label='Function', color='#e74a3b')
                ax_s.legend()
                st.pyplot(fig_s)