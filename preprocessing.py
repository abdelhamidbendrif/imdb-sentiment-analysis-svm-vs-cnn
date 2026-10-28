"""The same text preparation used in the Kaggle training notebook."""

import html
import re
from pathlib import Path

import nltk
from nltk import pos_tag
from nltk.stem import WordNetLemmatizer

RESOURCE_DIR = Path(__file__).resolve().parent / "nltk_data"
nltk.data.path.insert(0, str(RESOURCE_DIR))
lemmatizer = WordNetLemmatizer()


def check_resources():
    """Check resources without downloading anything during a prediction."""
    try:
        pos_tag(["movie"])
        lemmatizer.lemmatize("movies", pos="n")
    except LookupError as error:
        raise RuntimeError(
            "NLTK resources are missing. Run: python setup_resources.py"
        ) from error


def clean_text(text):
    text = html.unescape(text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.lower()
    return re.sub(r"\s+", " ", text).strip()


def normalize_negation(text):
    text = text.replace("\u2019", "'")
    text = re.sub(r"\bcan't\b", "can not", text)
    text = re.sub(r"\bwon't\b", "will not", text)
    text = re.sub(r"\bshan't\b", "shall not", text)
    text = re.sub(r"\bcannot\b", "can not", text)
    return re.sub(r"n't\b", " not", text)


def tokenize_text(text):
    return re.findall(r"[a-z]+(?:'[a-z]+)?", text)


def lemmatize_tokens(tokens):
    tagged_tokens = pos_tag(tokens)
    pos_mapping = {"J": "a", "V": "v", "N": "n", "R": "r"}
    lemmas = []
    for word, tag in tagged_tokens:
        wordnet_pos = pos_mapping.get(tag[0])
        if wordnet_pos is not None:
            lemma = lemmatizer.lemmatize(word, pos=wordnet_pos)
        else:
            lemma = word
        lemmas.append(lemma)
    return lemmas


def preprocess_review(cleaned_text):
    """Accept already-cleaned text, exactly like the notebook function."""
    normalized = normalize_negation(cleaned_text)
    tokens = tokenize_text(normalized)
    return " ".join(lemmatize_tokens(tokens))


def preprocess_raw_review(review):
    if not isinstance(review, str) or not review.strip():
        raise ValueError("Please enter a non-empty English movie review.")
    processed = preprocess_review(clean_text(review))
    if not processed:
        raise ValueError("The review contains no usable words.")
    return processed
