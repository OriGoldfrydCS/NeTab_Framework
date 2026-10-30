"""Raw-record synthesis with optional replayable proposal plans."""

from dataclasses import dataclass
import numpy as np
from .selection import probabilities, categorical


@dataclass
class GenerationResult:
    rows: list
    selected_donors: np.ndarray
    selected_slots: np.ndarray
    probabilities: np.ndarray
    current: np.ndarray
    reference: np.ndarray


class Generator:
    def __init__(
        self, proposal, scorer=None, rule="relative", k=8, m=4, temperature=1.0, candidate_filter=None
    ):
        if rule not in ("relative", "uniform", "absolute"):
            raise ValueError("unknown selection rule")
        if rule != "uniform" and scorer is None:
            raise ValueError("scored selection needs a scorer")
        if (
            any(type(v) is not int or v < 1 for v in (k, m))
            or not np.isfinite(temperature)
            or temperature <= 0
        ):
            raise ValueError("invalid K, M or temperature")
        self.proposal, self.scorer, self.rule = proposal, scorer, rule
        self.k, self.m, self.temperature = k, m, temperature
        if candidate_filter is not None and not callable(candidate_filter):
            raise TypeError("candidate_filter must be callable")
        self.candidate_filter = candidate_filter

    def generate(self, label, n, seed=42, plan=None, dataset_id=1, class_index=0):
        if type(n) is not int or n < 1:
            raise ValueError("n must be a positive integer")
        p = self.proposal
        plan = plan or p.plan(label, n, seed, self.k, self.m, dataset_id, class_index)
        positions = p.schema.positions
        p.validate_plan(plan, label, n, self.k, self.m)
        histories = [[] for _ in range(n)]
        donors = np.full((n, positions), -1, np.int32)
        slots = np.full((n, positions), -1, np.int32)
        prob = np.full((n, positions, self.k), np.nan)
        current = np.full_like(prob, np.nan)
        reference = np.full((n, positions, self.m, self.k), np.nan)
        for j in range(positions):
            for ri in np.flatnonzero(plan.lengths > j):
                ids = plan.candidate_donors[ri, j]
                candidates = [p.rows[int(d)][j] for d in ids]
                allowed = np.array(
                    [
                        k
                        for k, candidate in enumerate(candidates)
                        if self.candidate_filter is None
                        or self.candidate_filter(tuple(histories[ri]), tuple(candidate))
                    ],
                    dtype=int,
                )
                if not len(allowed):
                    raise ValueError(f"candidate_filter rejected all candidates at row {ri}, position {j}")
                accepted = [candidates[k] for k in allowed]
                count = len(accepted)
                # At empty history, all relative scores are identically zero.
                if self.rule == "uniform" or (j == 0 and self.rule == "relative"):
                    pr = np.full(count, 1 / count)
                else:
                    queries = [(c, histories[ri], label) for c in accepted]
                    if self.rule == "relative":
                        for ref in range(self.m):
                            bank = plan.bank_donors[int(plan.lengths[ri]), ref]
                            h = [p.rows[int(bank[pos])][pos] for pos in range(j)]
                            queries.extend((c, h, label) for c in accepted)
                    values = np.asarray(
                        self.scorer.score_many(queries)
                        if hasattr(self.scorer, "score_many")
                        else [self.scorer.score(*query) for query in queries]
                    )
                    if values.shape != (len(queries),) or not np.isfinite(values).all():
                        raise ValueError("scorer must return one finite NLL per query")
                    current[ri, j, allowed] = values[:count]
                    if self.rule == "relative":
                        reference[ri, j][:, allowed] = values[count:].reshape(self.m, count)
                    pr = probabilities(
                        current[ri, j, allowed], reference[ri, j][:, allowed], self.rule, self.temperature
                    )
                full_pr = np.zeros(self.k)
                full_pr[allowed] = pr
                slot = categorical(full_pr, plan.selection_uniforms[ri, j])
                donors[ri, j], slots[ri, j], prob[ri, j] = ids[slot], slot, full_pr
                histories[ri].append(tuple(candidates[slot]))
        rows = [p.schema.pad(h) for h in histories]
        return GenerationResult(rows, donors, slots, prob, current, reference)
