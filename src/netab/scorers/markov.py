"""Smoothed previous-record and within-record count scorer."""

from collections import Counter
import math
from .base import Scorer


class MarkovScorer(Scorer):
    def __init__(self, quantizer, smoothing=0.5, backoff=10.0, vocabulary_size=None):
        if smoothing <= 0 or backoff <= 0:
            raise ValueError("smoothing and backoff must be positive")
        self.quantizer = quantizer
        self.smoothing, self.backoff = smoothing, backoff
        self.vocabulary_size = vocabulary_size or len(set(quantizer.vocabulary) | {"<UNK_VAL>"})
        self.base, self.context, self.bt, self.ct = (Counter() for _ in range(4))

    def fit(self, rows, labels, schema, train_ids=None):
        for counter in (self.base, self.context, self.bt, self.ct):
            counter.clear()
        ids = range(len(rows)) if train_ids is None else train_ids
        for rid in ids:
            previous = ("BOS",)
            for j, record in enumerate(rows[rid][: schema.length(rows[rid])]):
                q = self.quantizer(record)
                for f, v in enumerate(q):
                    bk = (labels[rid], j, f, q[:f])
                    ck = (labels[rid], j, f, previous, q[:f])
                    self.base[bk, v] += 1
                    self.bt[bk] += 1
                    self.context[ck, v] += 1
                    self.ct[ck] += 1
                previous = q
        return self

    def score(self, candidate, history, label):
        q = self.quantizer(candidate)
        previous = self.quantizer(history[-1]) if history else ("BOS",)
        losses = []
        for f, v in enumerate(q):
            bk = (label, len(history), f, q[:f])
            ck = (label, len(history), f, previous, q[:f])
            p0 = (self.base[bk, v] + self.smoothing) / (self.bt[bk] + self.smoothing * self.vocabulary_size)
            p = (self.context[ck, v] + self.backoff * p0) / (self.ct[ck] + self.backoff)
            losses.append(-math.log(p))
        return sum(losses) / len(losses)
