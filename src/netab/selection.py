"""Stable selection rules over candidate slots (duplicates retained)."""

import numpy as np


def softmax(scores, temperature=1.0):
    x = np.asarray(scores, dtype=float)
    if x.ndim != 1 or not len(x) or not np.isfinite(x).all():
        raise ValueError("finite nonempty score vector required")
    if not np.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be positive and finite")
    z = np.exp((x - x.max()) / temperature)
    return z / z.sum()


def probabilities(current, reference=None, rule="relative", temperature=1.0):
    if rule == "uniform":
        return np.full(len(current), 1 / len(current))
    if rule == "absolute":
        return softmax(-np.asarray(current), temperature)
    if rule == "relative":
        return softmax(np.asarray(reference).mean(axis=0) - current, temperature)
    raise ValueError("unknown selection rule")


def categorical(probability, u):
    p = np.asarray(probability)
    if not 0 <= u < 1 or not np.isfinite(p).all() or np.any(p < 0) or not np.isclose(p.sum(), 1):
        raise ValueError("invalid categorical input")
    return min(len(p) - 1, int(np.searchsorted(np.cumsum(p), u, side="right")))
