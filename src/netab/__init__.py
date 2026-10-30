"""NeTab: empirical records, configurable compatibility scoring."""

from .schema import Field, Schema
from .quantize import LogQuantizer, Quantizer
from .serialize import Serializer
from .proposal import EmpiricalProposal, ProposalPlan
from .generator import Generator, GenerationResult
from .scorers.markov import MarkovScorer
from .validate import validate_rows

__version__ = "0.1.0"
__all__ = [
    "Field",
    "Schema",
    "LogQuantizer",
    "Quantizer",
    "Serializer",
    "EmpiricalProposal",
    "ProposalPlan",
    "Generator",
    "GenerationResult",
    "MarkovScorer",
    "validate_rows",
]
