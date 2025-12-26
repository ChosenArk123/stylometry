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

# --- PAGE CONFIG ---
st.set_page_config(page_title="Stylometry Ultra", layout="wide", page_icon="🕵️")

# --- STYLING ---
st.markdown("""
<style>
    .metric-card {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #4e73df;
        margin-bottom: 10px;
    }
    .insight-text {
        font-size: 14px;
        color: #555;
        font-style: italic;
    }
    .verdict-box {
        padding: 20px;
        border-radius: 10px;
        background-color: #e8eaed;
        margin: 20px 0;
    }
</style>
""", unsafe_allow_html=True)

st.title("🕵️ Stylometry Analysis Suite")
st.markdown("### Forensic Authorship & Linguistic Profiling")

# --- SIDEBAR CONFIG ---
st.sidebar.header("Analysis Mode")
mode = st.sidebar.radio("Select Operation:", ["Single Text Analysis", "Comparative Analysis"])


# --- HELPER: INTERPRETATION ENGINE ---
def get_interpretations(stats):
    """
    Generates human-readable insights based on metric thresholds.
    """
    insights = {}

    # 1. Flesch-Kincaid Grade
    g = stats['grade']
    if g < 8:
        insights[
            'grade'] = "This text is **Conversational**. It uses simple words and short sentences, making it accessible to a wide audience (e.g., blog posts, dialogue)."
    elif g < 13:
        insights[
            'grade'] = "This text is **Standard**. It requires high-school level reading skills, typical of newspapers or mainstream non-fiction."
    else:
        insights[
            'grade'] = "This text is **Academic/Technical**. It requires university-level education to parse comfortably. Expect jargon and complex reasoning."

    # 2. Lexical Diversity (Type-Token Ratio)
    d = stats['diversity']
    if d < 0.45:
        insights[
            'diversity'] = "The vocabulary is **Repetitive**. The author reuses the same words frequently, which is common in technical manuals, legal contracts, or novice writing."
    elif d < 0.65:
        insights[
            'diversity'] = "The vocabulary is **Balanced**. The author varies their word choice enough to maintain interest without being flowery."
    else:
        insights[
            'diversity'] = "The vocabulary is **Rich/Descriptive**. The author uses a vast array of unique words, typical of literary fiction, poetry, or highly expressive essays."

    # 3. Syntactic Depth
    depth = stats['depth']
    if depth < 3.5:
        insights[
            'depth'] = "Sentence structures are **Simple**. The author prefers direct Subject-Verb-Object constructions (e.g., 'The dog ran.')."
    elif depth < 5.5:
        insights[
            'depth'] = "Sentence structures are **Standard**. The text uses dependent clauses and conjunctions naturally."
    else:
        insights[
            'depth'] = "Sentence structures are **Complex**. The author heavily nests clauses (e.g., 'The dog, which was brown, ran...'). This indicates a formal or older writing style."

    # 4. Function Ratio
    f_ratio = stats['func_ratio']
    if f_ratio > 0.55:
        insights[
            'func'] = "The text is **Glue-Heavy**. It relies on function words (the, of, and, to) to connect ideas, often signaling a spoken or 'soft' tone."
    else:
        insights[
            'func'] = "The text is **Content-Dense**. It packs many nouns and verbs into few words, signaling a 'hard' or information-heavy tone."

    return insights


# --- HELPER: ANALYSIS WRAPPER ---
def analyze_text(file_obj):
    with tempfile.NamedTemporaryFile(delete=False, suffix=Path(file_obj.name).suffix) as tmp:
        tmp.write(file_obj.getvalue())
        tmp_path = tmp.name

    try:
        text = load_text(tmp_path)
    finally:
        os.unlink(tmp_path)

    doc, sentences, tokens = preprocess(text)
    s_stats = sentence_stats(sentences)
    ratios = function_content_ratio(tokens)
    grade = flesch_kincaid_grade(tokens, sentences)
    diversity = lexical_diversity(tokens)
    depth = average_syntactic_depth(sentences)
    raw_lens = sentence_lengths(sentences)

    top_bigrams = get_top_ngrams(tokens, n=2, limit=10)
    all_bigrams = [f"{x[0]} {x[1]}" for x in zip([t.text for t in tokens], [t.text for t in tokens][1:])]

    vector = np.array([
        s_stats['mean_sentence_length'],
        diversity,
        depth,
        ratios['function_ratio'],
        grade
    ])

    return {
        "filename": file_obj.name,
        "text_preview": text[:500] + "...",
        "stats": {
            "mean_len": s_stats['mean_sentence_length'],
            "std_len": s_stats['std_sentence_length'],
            "max_len": s_stats['max_sentence_length'],
            "grade": grade,
            "diversity": diversity,
            "depth": depth,
            "func_ratio": ratios['function_ratio'],
            "content_ratio": ratios['content_ratio'],
            "raw_lens": raw_lens,
        },
        "top_bigrams": top_bigrams,
        "all_bigrams": all_bigrams,
        "vector": vector
    }


# --- MODE 1: SINGLE ANALYSIS ---
if mode == "Single Text Analysis":
    st.sidebar.info("Upload a text to see a deep-dive breakdown of its linguistic structure.")
    uploaded = st.file_uploader("Upload Document (PDF, TXT, DOCX)", type=['txt', 'pdf', 'docx'])

    if uploaded:
        with st.spinner("Analyzing linguistics..."):
            data = analyze_text(uploaded)
            stats = data['stats']
            insights = get_interpretations(stats)

        st.header(f"📄 Report: {data['filename']}")

        # --- METRIC CARDS WITH EXPLANATIONS ---
        col1, col2 = st.columns(2)

        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <h3>📖 Reading Grade: {stats['grade']:.1f}</h3>
                <p class="insight-text">{insights['grade']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <h3>🧠 Syntactic Depth: {stats['depth']:.2f}</h3>
                <p class="insight-text">{insights['depth']}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <h3>🎨 Lexical Diversity: {stats['diversity']:.3f}</h3>
                <p class="insight-text">{insights['diversity']}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown(f"""
            <div class="metric-card">
                <h3>⚖️ Content Density: {stats['content_ratio']:.2f}</h3>
                <p class="insight-text">{insights['func']}</p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("---")

        # --- VISUALIZATIONS ---
        col_left, col_right = st.columns(2)

        with col_left:
            st.subheader("1. Sentence Rhythm (Pacing)")
            st.markdown("""
            **What this shows:** A "spiky" graph means the author mixes short and long sentences (dynamic pacing). 
            A "flat" bell curve means the author is very consistent (monotone pacing).
            """)
            fig1, ax1 = plt.subplots(figsize=(6, 4))
            sns.histplot(stats['raw_lens'], kde=True, bins=20, color="skyblue", ax=ax1)
            ax1.set_xlabel("Words per Sentence")
            st.pyplot(fig1)

        with col_right:
            st.subheader("2. Vocabulary Architecture")
            st.markdown("""
            **What this shows:** * **Blue (Content):** Nouns, Verbs, Adjectives (The meat of the sentence).
            * **Red (Function):** Prepositions, Articles, Pronouns (The glue).
            """)
            fig2, ax2 = plt.subplots(figsize=(6, 4))
            labels = ['Function (Glue)', 'Content (Info)']
            sizes = [stats['func_ratio'], stats['content_ratio']]
            colors = ['#ff9999', '#66b3ff']
            ax2.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90, colors=colors)
            st.pyplot(fig2)

        st.markdown("---")

        st.subheader("3. Behavioral Fingerprint (Top Phrases)")
        st.write(
            "These are the author's subconscious 'go-to' phrases. High frequency of unique bigrams (like 'it is' or 'in the') creates a recognizable signature.")
        if data['top_bigrams']:
            phrases, counts = zip(*data['top_bigrams'])
            fig3, ax3 = plt.subplots(figsize=(10, 3))
            y_pos = np.arange(len(phrases))
            ax3.barh(y_pos, counts, align='center', color='teal')
            ax3.set_yticks(y_pos)
            ax3.set_yticklabels(phrases)
            ax3.invert_yaxis()
            st.pyplot(fig3)
        else:
            st.write("Not enough text for bigrams.")

# --- MODE 2: COMPARATIVE ANALYSIS ---
elif mode == "Comparative Analysis":
    st.sidebar.info("Upload two texts to see if they were written by the same person.")
    c1, c2 = st.columns(2)
    f1 = c1.file_uploader("Reference Text A", type=['txt', 'pdf', 'docx'])
    f2 = c2.file_uploader("Candidate Text B", type=['txt', 'pdf', 'docx'])

    if f1 and f2:
        if st.button("🚀 Run Forensic Comparison"):
            with st.spinner("Analyzing vectors..."):
                d1 = analyze_text(f1)
                d2 = analyze_text(f2)
                cos_sim = cosine_similarity(d1['vector'], d2['vector'])
                jac_sim = jaccard_similarity(d1['all_bigrams'], d2['all_bigrams'])

            # --- VERDICT SECTION ---
            st.markdown("<div class='verdict-box'>", unsafe_allow_html=True)
            st.subheader("🏛️ The Verdict")

            col_v1, col_v2 = st.columns(2)
            col_v1.metric("Structural Match (Cosine)", f"{cos_sim:.4f}")
            col_v2.metric("Phrase Overlap (Jaccard)", f"{jac_sim:.4f}")

            # detailed interpretation of the verdict
            if cos_sim > 0.95:
                st.success(
                    "✅ **Identical Structure.** The 'skeleton' of these texts (sentence length, complexity, rhythm) is indistinguishable.")
            elif cos_sim > 0.85:
                st.info(
                    "ℹ️ **Similar Structure.** These texts likely share a genre or education level, but show minor stylistic differences.")
            else:
                st.error(
                    "❌ **Distinct Structure.** These texts are fundamentally different in how they are constructed.")

            if jac_sim > 0.20:
                st.warning(
                    "⚠️ **High Phrase Overlap.** The authors use the exact same word pairings significantly often. This is a strong marker of single authorship or copy-pasting.")

            st.markdown("</div>", unsafe_allow_html=True)

            # --- VISUALIZATION SECTION ---
            st.subheader("📊 Comparative Graphs")

            # Graph 1
            st.markdown("#### 1. Sentence Rhythm Overlap")
            st.caption("Do the authors 'breathe' at the same rate? Overlapping curves indicate similar pacing.")
            fig_rhythm, ax_r = plt.subplots(figsize=(10, 4))
            sns.histplot(d1['stats']['raw_lens'], color="blue", label=d1['filename'], kde=True, stat="density",
                         alpha=0.4, ax=ax_r)
            sns.histplot(d2['stats']['raw_lens'], color="orange", label=d2['filename'], kde=True, stat="density",
                         alpha=0.4, ax=ax_r)
            ax_r.legend()
            st.pyplot(fig_rhythm)

            col_a, col_b = st.columns(2)

            with col_a:
                # Graph 2
                st.markdown("#### 2. Complexity Fingerprint")
                st.caption("Comparing the 'difficulty' and 'richness' of the texts.")
                labels = ['Grade', 'Diversity (x10)', 'Depth', 'Func Ratio (x10)']
                v1 = [d1['stats']['grade'], d1['stats']['diversity'] * 10, d1['stats']['depth'],
                      d1['stats']['func_ratio'] * 10]
                v2 = [d2['stats']['grade'], d2['stats']['diversity'] * 10, d2['stats']['depth'],
                      d2['stats']['func_ratio'] * 10]

                x = np.arange(len(labels))
                width = 0.35
                fig_bar, ax_b = plt.subplots()
                ax_b.bar(x - width / 2, v1, width, label='Text A', color='blue')
                ax_b.bar(x + width / 2, v2, width, label='Text B', color='orange')
                ax_b.set_xticks(x)
                ax_b.set_xticklabels(labels)
                ax_b.legend()
                st.pyplot(fig_bar)

            with col_b:
                # Graph 3
                st.markdown("#### 3. Vocabulary Architecture")
                st.caption("Is one text more 'fluff' (Function words) and the other more 'facts' (Content words)?")
                labels = ['Text A', 'Text B']
                func_vals = [d1['stats']['func_ratio'], d2['stats']['func_ratio']]
                cont_vals = [d1['stats']['content_ratio'], d2['stats']['content_ratio']]

                fig_stack, ax_s = plt.subplots()
                ax_s.bar(labels, cont_vals, label='Content', color='#66b3ff')
                ax_s.bar(labels, func_vals, bottom=cont_vals, label='Function', color='#ff9999')
                ax_s.legend()
                st.pyplot(fig_stack)