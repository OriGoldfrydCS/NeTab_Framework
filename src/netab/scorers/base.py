"""Lower score means more probable candidate values."""

from abc import ABC, abstractmethod


class Scorer(ABC):
    @abstractmethod
    def score(self, candidate, history, label):
        """Return S_θ(c | h, a): candidate c, raw history h, class a.

        API names candidate, history, and label correspond to c, h, and a.
        The loss is averaged over candidate value tokens only.
        """
        raise NotImplementedError

    def score_many(self, queries):
        """Optional batching hook; default preserves query order."""
        return [self.score(candidate, history, label) for candidate, history, label in queries]
