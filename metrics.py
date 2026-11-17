"""Scoring metrics for evaluating LLM prompt outputs against expected ground truth."""
import re
from difflib import SequenceMatcher
from typing import List, Dict


def normalize_text(text: str) -> str:
    """Lowercases, strips punctuation, and collapses whitespace."""
    cleaned = re.sub(r"[^a-z0-9\s]", "", text.lower())
    return " ".join(cleaned.split())


def exact_match_score(predicted: str, expected: str) -> float:
    """Returns 1.0 if normalized strings match identically, else 0.0."""
    return 1.0 if normalize_text(predicted) == normalize_text(expected) else 0.0


def token_f1_score(predicted: str, expected: str) -> float:
    """Computes token-level F1 overlap between predicted and expected answers."""
    pred_tokens = normalize_text(predicted).split()
    exp_tokens = normalize_text(expected).split()

    if not pred_tokens and not exp_tokens:
        return 1.0
    if not pred_tokens or not exp_tokens:
        return 0.0

    common = {}
    for tok in pred_tokens:
        if tok in exp_tokens:
            common[tok] = min(pred_tokens.count(tok), exp_tokens.count(tok))

    num_same = sum(common.values())
    if num_same == 0:
        return 0.0

    precision = num_same / len(pred_tokens)
    recall = num_same / len(exp_tokens)
    return round((2 * precision * recall) / (precision + recall), 4)


def similarity_score(predicted: str, expected: str) -> float:
    """Computes character-sequence similarity ratio (0.0 to 1.0)."""
    ratio = SequenceMatcher(None, normalize_text(predicted), normalize_text(expected)).ratio()
    return round(ratio, 4)


def evaluate_prediction(predicted: str, expected: str, threshold: float = 0.75) -> Dict[str, object]:
    """Combines Exact Match, Token F1, and Similarity into a composite accuracy verdict."""
    em = exact_match_score(predicted, expected)
    f1 = token_f1_score(predicted, expected)
    sim = similarity_score(predicted, expected)
    composite = round((0.4 * em) + (0.4 * f1) + (0.2 * sim), 4)
    passed = (em == 1.0) or (f1 >= threshold) or (composite >= threshold)

    return {
        "exact_match": em,
        "token_f1": f1,
        "similarity": sim,
        "composite_score": composite,
        "passed": passed
    }
