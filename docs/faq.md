# FAQ

## Can NeTab generate a record value that never appears in my data?

No. Each whole record is copied from a real fitting row of the requested class
at the same position. Output therefore consists of records already observed
in the fitting data, with the configured null record used for padding.
New sequences of those records can be generated. Quantization affects scoring
only and does not create new output values.

## Does NeTab enforce constraints between records (e.g., increasing timestamps)?

Such constraints are not guaranteed by construction. A fitted scorer learns
preferences from the history and can favor compatible candidates, but a
preference is not a hard constraint. To enforce one, pass
`candidate_filter(history, candidate) -> bool` to `Generator`:

```python
from netab import Field, Schema, EmpiricalProposal, Generator

schema = Schema(positions=2, fields=[Field(name="timestamp", dtype="int")], null_record=(0,))
rows = [schema.pad(records=[(1,), (3,)]), schema.pad(records=[(2,), (4,)])]
proposal = EmpiricalProposal(schema=schema, rows=rows, labels=["A", "A"])


def increasing(history, candidate):
    return not history or float(candidate[0]) > float(history[-1][0])


generator = Generator(proposal=proposal, rule="uniform", candidate_filter=increasing)
result = generator.generate(label="A", n=8, seed=42)
```

The filter receives an immutable raw-history tuple and a raw candidate tuple.
It runs before scoring, including for uniform selection. Rejected slots are
never scored and receive zero selection probability; remaining slots are
renormalized. Relative scoring uses the same accepted candidates for the current
and reference histories. To combine a filter with a fitted scorer, pass
`scorer=scorer` and use `rule="relative"` or `rule="absolute"`.

If all `K` sampled candidates are rejected, generation raises `ValueError`
with the row and position. It does not relax the constraint, resample, or return
a partially generated result. This can occur even when an acceptable donor
exists outside the sampled slots. Local constraints may also lead to a later
dead end; the filter does not perform lookahead or enforce global feasibility.

## Can two candidate slots be identical?

Yes. Sampling is with replacement. Duplicate slots remain separate; their
probabilities add when considering the probability of a particular raw record.

## Why is the first relative selection uniform?

Its current history and all reference histories are empty. Their scores cancel
exactly. Absolute selection instead scores the class-conditioned candidate
without a reference correction. With a candidate filter, relative selection
is uniform over the accepted slots.

## What does validation mean?

V1 means an active prefix with exact null padding. V2 means each active record
satisfies the configured record domain. These checks do not enforce relations
between records. For example, monotone times require application-level logic.

## Is zero timing treated as padding?

Only the activity predicate defines padding. The traffic example uses direction,
so zero timing in an active record is valid. Every field of an inactive record
must equal its corresponding null value.

## Is this a privacy mechanism?

No formal privacy guarantee is provided. The method copies empirical records
and may reproduce entire fitting rows. Evaluate privacy requirements separately
for the intended application.

## Does the library need a GPU?

No. Markov runs on CPU. Neural scorers and training default to `device="auto"`,
which selects CUDA when available and otherwise CPU. Use `device="cpu"` in the
Python API or `--device cpu` in the CLI to require CPU. Large causal models can
exceed available memory; start with Markov or a small Tiny configuration.
Thread counts follow PyTorch’s current setting by default. Use CLI `--threads N`
or the trainer’s `num_threads=N` to set an explicit positive count, without a
fixed upper limit.

## Why can neural probabilities change with the same seed?

A seed controls sampling, but floating-point reductions can depend on physical
batch shape, backend, and package versions. Fix those settings for numerical
comparisons. The sampler uses an explicit inverse CDF with right-sided boundary
handling. Small probability changes can affect choices near CDF boundaries.
