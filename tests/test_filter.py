import numpy as np
import pytest
from netab import Schema, Field, EmpiricalProposal, Generator
from netab.scorers.base import Scorer


class OddOnlyScorer(Scorer):
    def __init__(self):
        self.seen = []

    def score(self, candidate, history, label):
        assert candidate[0] % 2 == 1, "rejected candidate reached scoring"
        self.seen.append(candidate)
        return float(candidate[0]) / 10


@pytest.mark.parametrize("rule", ["relative", "absolute", "uniform"])
def test_filter_runs_before_scoring_preserves_slots_and_constraints(rule):
    schema = Schema(positions=2, fields=[Field("timestamp", "int")], null_record=(0,))
    rows = [[(1,), (3,)], [(2,), (4,)], [(3,), (5,)], [(4,), (6,)]]
    proposal = EmpiricalProposal(schema, rows, ["A"] * 4)
    plan = proposal.plan("A", 8, k=4, m=2)
    plan.candidate_donors[:] = np.arange(4)

    def valid(history, candidate):
        assert isinstance(history, tuple) and isinstance(candidate, tuple)
        return candidate[0] % 2 == 1 and (not history or candidate[0] > history[-1][0])

    scorer = OddOnlyScorer()
    result = Generator(proposal, scorer, rule=rule, k=4, m=2, candidate_filter=valid).generate(
        "A", 8, plan=plan
    )
    for ri, row in enumerate(result.rows):
        assert row[0][0] < row[1][0] and all(record[0] % 2 for record in row)
        for j in range(2):
            for slot, donor in enumerate(plan.candidate_donors[ri, j]):
                if not valid(tuple(row[:j]), rows[donor][j]):
                    assert result.probabilities[ri, j, slot] == 0
            chosen = result.selected_slots[ri, j]
            assert result.selected_donors[ri, j] == plan.candidate_donors[ri, j, chosen]
            assert result.probabilities[ri, j].sum() == pytest.approx(1)
    assert bool(scorer.seen) == (rule != "uniform")


def test_filter_all_rejected_fails_without_scoring_or_fallback():
    schema = Schema(1, [Field("x", "int")], (0,))
    proposal = EmpiricalProposal(schema, [[(2,)]], ["A"])
    scorer = OddOnlyScorer()
    with pytest.raises(ValueError, match="row 0, position 0"):
        Generator(proposal, scorer, candidate_filter=lambda history, candidate: False).generate("A", 1)
    assert not scorer.seen


def test_accept_all_filter_preserves_unfiltered_probabilities_and_choices():
    schema = Schema(2, [Field("x", "int")], (0,))
    proposal = EmpiricalProposal(schema, [[(1,), (3,)], [(3,), (5,)]], ["A", "A"])
    scorer = OddOnlyScorer()
    unfiltered = Generator(proposal, scorer).generate("A", 10)
    filtered = Generator(proposal, scorer, candidate_filter=lambda h, c: True).generate("A", 10)
    assert filtered.rows == unfiltered.rows
    assert np.array_equal(filtered.probabilities, unfiltered.probabilities, equal_nan=True)
