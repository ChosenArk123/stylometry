import numpy as np
from collections import Counter

def sentence_lengths(sentences):
    return np.array([len(sent) for sent in sentences])


def sentence_stats(sentences):
    lengths = sentence_lengths(sentences)
    if len(lengths) == 0:
        return {"mean_sentence_length": 0, "std_sentence_length": 0, "min_sentence_length": 0, "max_sentence_length": 0}
    return {
        "mean_sentence_length": lengths.mean(),
        "std_sentence_length": lengths.std(),
        "min_sentence_length": lengths.min(),
        "max_sentence_length": lengths.max(),
    }


def get_ngrams(tokens, n=2):
    """
    Generates a list of n-grams (tuples of n words) from the token list.
    We KEEP stop words because they are stylistically significant.
    """
    # We only take alpha characters, but we keep "the", "and", etc.
    words = [t.text.lower() for t in tokens if t.is_alpha]

    # Zip the list against itself shifted by 1 to create pairs
    # Example: [a, b, c, d] -> [(a,b), (b,c), (c,d)]
    return [" ".join(words[i:i + n]) for i in range(len(words) - n + 1)]


def get_top_ngrams(tokens, n=2, limit=10):
    """Returns the most frequent n-grams and their counts."""
    ngrams = get_ngrams(tokens, n)
    if not ngrams: return []
    return Counter(ngrams).most_common(limit)


def jaccard_similarity(list_a, list_b):
    """
    Calculates intersection over union.
    0.0 = No shared phrases.
    1.0 = Identical set of phrases.
    """
    set_a = set(list_a)
    set_b = set(list_b)

    intersection = len(set_a.intersection(set_b))
    union = len(set_a.union(set_b))

    return intersection / union if union > 0 else 0

def lexical_diversity(tokens):
    words = [t.text.lower() for t in tokens if t.is_alpha]
    unique_words = set(words)
    return len(unique_words) / len(words) if words else 0


def sentence_depth(sent):
    # Calculate the maximum depth of the syntactic tree
    return max([len(list(token.ancestors)) for token in sent], default=0)


def average_syntactic_depth(sentences):
    depths = [sentence_depth(sent) for sent in sentences]
    return sum(depths) / len(depths) if depths else 0


def function_content_ratio(tokens):
    function_pos = {"DET", "ADP", "AUX", "CCONJ", "SCONJ", "PART", "PRON"}
    function_count = 0
    content_count = 0

    for token in tokens:
        if not token.is_alpha:
            continue
        if token.pos_ in function_pos:
            function_count += 1
        else:
            content_count += 1

    total = function_count + content_count
    if total == 0:
        return {"function_ratio": 0, "content_ratio": 0}

    return {
        "function_ratio": function_count / total,
        "content_ratio": content_count / total
    }


def count_syllables(word):
    word = word.lower()
    count = 0
    vowels = "aeiouy"
    if len(word) == 0: return 0
    if word[0] in vowels: count += 1
    for index in range(1, len(word)):
        if word[index] in vowels and word[index - 1] not in vowels:
            count += 1
    if word.endswith("e"): count -= 1
    if count == 0: count += 1
    return count


def flesch_kincaid_grade(tokens, sentences):
    words = [t.text for t in tokens if t.is_alpha]
    if not words or not sentences: return 0

    total_words = len(words)
    total_sentences = len(sentences)
    total_syllables = sum(count_syllables(w) for w in words)

    return 0.39 * (total_words / total_sentences) + 11.8 * (total_syllables / total_words) - 15.59


def get_stylometric_vector(doc_stats):
    # Converts dictionary stats into a numpy array for comparison
    return np.array([
        doc_stats['mean_sentence_length'],
        doc_stats['lexical_diversity'],
        doc_stats['average_syntactic_depth'],
        doc_stats['function_ratio'],
        doc_stats['readability_grade']
    ])


def cosine_similarity(vec1, vec2):
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    if norm1 == 0 or norm2 == 0: return 0.0
    return np.dot(vec1, vec2) / (norm1 * norm2)