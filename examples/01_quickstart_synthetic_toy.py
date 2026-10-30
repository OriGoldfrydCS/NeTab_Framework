"""Generate and validate toy rows entirely in memory, on CPU."""

from netab import EmpiricalProposal, Generator, MarkovScorer, Quantizer, validate_rows
from netab.cli import toy

schema, rows, labels = toy()
proposal = EmpiricalProposal(schema, rows, labels)
scorer = MarkovScorer(Quantizer(schema)).fit(rows, labels, schema)
result = Generator(proposal, scorer).generate("demo", n=16, seed=42)
assert validate_rows(schema, result.rows)
print(f"Generated {len(result.rows)} valid rows on CPU.")
