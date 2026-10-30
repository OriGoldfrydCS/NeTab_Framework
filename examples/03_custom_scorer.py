"""A deterministic scorer plug-in illustrating the interface."""

import math
from netab import EmpiricalProposal, Generator
from netab.cli import toy
from netab.scorers.base import Scorer


class IndependentScorer(Scorer):
    def score(self, candidate, history, label):
        # NLL for an illustrative independent geometric amount model.
        return float(candidate[0]) * math.log(2)


schema, rows, labels = toy()
result = Generator(EmpiricalProposal(schema, rows, labels), IndependentScorer()).generate("demo", 8)
assert len(result.rows) == 8
print("Custom scorer generated valid records.")
