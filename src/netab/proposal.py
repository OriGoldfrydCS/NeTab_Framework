"""Empirical lengths and position-conditional donor pools."""

from dataclasses import dataclass
import numpy as np


@dataclass
class ProposalPlan:
    lengths: np.ndarray
    length_donors: np.ndarray
    candidate_donors: np.ndarray
    selection_uniforms: np.ndarray
    bank_donors: np.ndarray


class EmpiricalProposal:
    def __init__(self, schema, rows, labels, train_ids=None):
        if len(rows) != len(labels):
            raise ValueError("rows and labels differ in length")
        self.schema, self.rows, self.labels = schema, rows, list(labels)
        self.lengths = np.array([schema.length(r) for r in rows])
        self.train_ids = np.array(list(range(len(rows))) if train_ids is None else train_ids, dtype=int)
        if (
            len(set(self.train_ids)) != len(self.train_ids)
            or np.any(self.train_ids < 0)
            or np.any(self.train_ids >= len(rows))
        ):
            raise ValueError("invalid or duplicate training IDs")

    def donors(self, label, position=None):
        ids = np.array([i for i in self.train_ids if self.labels[i] == label], dtype=int)
        return ids if position is None else ids[self.lengths[ids] > position]

    def plan(self, label, n, seed=42, k=8, m=4, dataset_id=1, class_index=0):
        from .references import reference_bank

        if any(type(v) is not int or v < 0 for v in (seed, dataset_id, class_index)):
            raise ValueError("seed and stream indices must be nonnegative integers")
        if any(type(v) is not int or v < 1 for v in (n, k, m)):
            raise ValueError("n, K and M must be positive")
        ids = self.donors(label)
        if not len(ids):
            raise ValueError("class has no fitting donors")

        def rng(stream):
            return np.random.default_rng(
                np.random.SeedSequence([2026091708, seed, class_index, stream, dataset_id])
            )

        ld = rng(1).choice(ids, n)
        lengths = self.lengths[ld]
        u = rng(4).random((n, self.schema.positions))
        candidates = np.full((n, self.schema.positions, k), -1, np.int32)
        cr = rng(2)
        for j in range(self.schema.positions):
            active = np.flatnonzero(lengths > j)
            if len(active):
                candidates[active, j] = cr.choice(self.donors(label, j), (len(active), k))
        bank = reference_bank(self, label, lengths, seed, m, dataset_id, class_index)
        return ProposalPlan(lengths, ld, candidates, u, bank)

    def validate_plan(self, plan, label, n, k, m):
        p = self.schema.positions
        shapes = {
            "lengths": (n,),
            "length_donors": (n,),
            "candidate_donors": (n, p, k),
            "selection_uniforms": (n, p),
            "bank_donors": (p + 1, m, p),
        }
        for name, shape in shapes.items():
            a = np.asarray(getattr(plan, name))
            if a.shape != shape:
                raise ValueError(f"plan {name} has incorrect shape")
            if name != "selection_uniforms" and not np.issubdtype(a.dtype, np.integer):
                raise ValueError(f"plan {name} must contain integers")
        u = plan.selection_uniforms
        if not np.isfinite(u).all() or np.any(u < 0) or np.any(u >= 1):
            raise ValueError("plan uniforms must lie in [0, 1)")
        if not np.isin(plan.length_donors, self.donors(label)).all():
            raise ValueError("plan contains ineligible length donors")
        if not np.array_equal(plan.lengths, self.lengths[plan.length_donors]):
            raise ValueError("plan lengths differ from length donors")
        for j in range(p):
            active = plan.lengths > j
            eligible = self.donors(label, j)
            if not np.isin(plan.candidate_donors[active, j], eligible).all():
                raise ValueError("plan contains ineligible candidate donors")
            if np.any(plan.candidate_donors[~active, j] != -1):
                raise ValueError("inactive candidate positions must be -1")
            for length in np.unique(plan.lengths):
                if length > j and not np.isin(plan.bank_donors[length, :, j], eligible).all():
                    raise ValueError("plan contains ineligible reference donors")
