"""Diversity (within one world) and consistency (across repeated worlds).

Embeddings are injectable (`embed: list[str] -> (n, d) array`) so the maths
can be tested without downloading a model.
"""
from itertools import combinations
from math import log
from typing import Callable

import numpy as np

from .generate import World
from .prompts import DOMAINS

Embed = Callable[[list[str]], np.ndarray]
_st_model = None


def default_embed(texts: list[str]) -> np.ndarray:
    global _st_model
    if _st_model is None:
        from sentence_transformers import SentenceTransformer
        _st_model = SentenceTransformer("all-MiniLM-L6-v2")
    return _st_model.encode(texts, normalize_embeddings=True)


def _unit(x: np.ndarray) -> np.ndarray:
    return x / np.clip(np.linalg.norm(x, axis=1, keepdims=True), 1e-9, None)


# ---- Diversity: how different are the consequences inside ONE world? -------
def distinct_n(texts: list[str], n: int = 2) -> float:
    """Unique n-grams / total n-grams (lexical diversity)."""
    grams = []
    for t in texts:
        toks = t.lower().split()
        grams += [tuple(toks[i:i + n]) for i in range(len(toks) - n + 1)]
    return len(set(grams)) / len(grams) if grams else 0.0


def semantic_diversity(texts: list[str], embed: Embed = default_embed) -> float:
    """Mean pairwise cosine distance between consequences (0 = identical)."""
    if len(texts) < 2:
        return 0.0
    e = _unit(embed(texts))
    sims = [float(e[i] @ e[j]) for i, j in combinations(range(len(e)), 2)]
    return 1.0 - float(np.mean(sims))


def domain_coverage(texts: list[str], embed: Embed = default_embed) -> tuple[float, float]:
    """Assign each consequence to its nearest domain.

    Returns (fraction of domains hit, normalised entropy of the assignment).
    Measured by embedding so it applies to untagged (generic) output too.
    """
    if not texts:
        return 0.0, 0.0
    d = _unit(embed([f"{x.name}: {x.hint}" for x in DOMAINS]))
    t = _unit(embed(texts))
    assign = np.argmax(t @ d.T, axis=1)
    counts = np.bincount(assign, minlength=len(DOMAINS))
    p = counts[counts > 0] / counts.sum()
    entropy = -float(np.sum(p * np.log(p))) / log(len(DOMAINS))
    return float((counts > 0).mean()), entropy


# ---- Consistency: do repeated runs on the SAME rule agree? -----------------
def _best_match(a: np.ndarray, b: np.ndarray) -> float:
    s = a @ b.T
    return float((s.max(axis=1).mean() + s.max(axis=0).mean()) / 2)


def consistency(worlds: list[World], embed: Embed = default_embed) -> float:
    """Mean pairwise best-match cosine similarity between worlds.

    High = repeated runs produce the same themes. Low = every run is different.
    """
    embs = [_unit(embed(w.texts)) for w in worlds if w.texts]
    if len(embs) < 2:
        return float("nan")
    return float(np.mean([_best_match(a, b) for a, b in combinations(embs, 2)]))
