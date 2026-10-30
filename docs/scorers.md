# Scorers

A scorer receives a raw candidate record, a list of previous raw records, and a
class label. We write these as candidate `c`, history `h`, and class `a`,
with score `S_θ(c | h, a)`. Its return value is a finite mean negative log likelihood (NLL)
for the candidate's value tokens. Lower NLL means greater compatibility.
Class labels, delimiters, and the existing history condition the score but do
not contribute target losses to it.

| Scorer | Size (parameters) | GPU required | Relative speed | When to use |
| --- | --- | --- | --- | --- |
| Markov | Sparse counts; no neural parameters | No | Usually fastest | Local record dependencies with lightweight fitting |
| Tiny Transformer | Default `1,580,032 + 256(V + T)`; tied embeddings | Optional | Moderate | Train a compact causal model for a schema |
| Hugging Face causal LM | Model- and vocabulary-dependent | Recommended | Model-dependent | Use a pretrained or randomly initialized causal model |

`V` denotes model vocabulary size and `T` context capacity. The formula for Tiny
counts the code's default architecture. Obtain an instantiated neural model's
exact size with `sum(p.numel() for p in scorer.model.parameters())`; embedding
and output weights share storage in Tiny. Speed descriptions are qualitative,
not measurements. Neural scorers accept `device="auto"`, `"cpu"`, or a CUDA device; auto selects
CUDA if available and otherwise CPU.

## MarkovScorer

```python
from netab import MarkovScorer, Quantizer

scorer = MarkovScorer(Quantizer(schema), smoothing=0.5, backoff=10.0)
scorer.fit(rows, labels, schema)
```

For position `j`, field `f`, quantized prefix `z` of the current record, and
previous quantized record `q_prev`, counts are kept for a class-specific base
context `(j, f, z)` and a fuller context `(j, f, q_prev, z)`. The scorer receives
history `h` and extracts its last record for this context. The first record uses
an explicit beginning-of-sequence context.

```text
p0(v) = (base_count(v) + smoothing) / (base_total + smoothing * V)
p(v)  = (context_count(v) + backoff * p0(v)) / (context_total + backoff)
S_θ(c | h, a) = mean_field -log p(value)
```

The vocabulary size defaults to the union of quantizer symbols plus an unknown
value symbol. An explicit `vocabulary_size` can be supplied to match an
application's fixed vocabulary. Every field produces one quantized symbol.
Calling `fit` clears previously fitted counts.

## TinyTransformerScorer

```python
import torch
from netab import Serializer, Quantizer
from netab.scorers.tiny import TinyTransformerScorer

torch.manual_seed(42)
scorer = TinyTransformerScorer(Serializer(schema, Quantizer(schema)), classes=sorted(set(labels)))
```

Defaults are two causal encoder layers, width 256, four heads, a feed-forward
width four times the embedding width, and dropout 0.1. Token and output
embeddings are tied. Each field value and structural piece is one symbol;
classes must be registered before training. Context capacity is derived from
the schema's maximum serialized row length.

The model is randomly initialized. Train it before scored generation. The
trainer restores the selected validation checkpoint in memory. Python users
loading a checkpoint must recreate the same schema, class order, quantizers,
and architecture. The CLI stores the Tiny class order and architecture with
its checkpoint directory.

## HFCausalLMScorer

Install `.[hf]`, then construct from a model name or local model directory:

```python
from netab import Serializer, Quantizer
from netab.scorers.hf import HFCausalLMScorer

scorer = HFCausalLMScorer.from_pretrained(
    "gpt2",
    Serializer(schema, Quantizer(schema)),
    from_scratch=False,
    extend_tokens=True,
)
```

`from_scratch=True` uses the selected architecture with randomly initialized
weights. The default loads pretrained weights. Tokenizer extensions add schema
delimiters, all quantizer symbols, and an unknown-value symbol; model embeddings
are resized to the resulting vocabulary. A padding symbol is added if needed.
Use `local_files_only=True` when all model files must already be present.
No custom remote model code is enabled by this wrapper.

The wrapper uses a fast tokenizer's character offsets to track **all** tokens
for each candidate field. A token that crosses a value boundary is rejected.
There is no silent context truncation. The full model vocabulary `|V|`, including
structural and class tokens, participates in each token's log-softmax.
A value represented by several tokens contributes all of those token losses:

```text
S_default = sum_field sum_token NLL(token) / sum_field token_count(field)
S_per_value = mean_field [sum_token NLL(token) / token_count(field)]
```

`normalize_per_value=True` enables the second expression. It is disabled by
default. Neither expression renormalizes probabilities over value symbols
alone. When each value is one token, the two expressions coincide.

`scoring_batch_size` controls physical batch size, and `fixed_length` optionally
sets right-padding width. Partial batches are filled with duplicate inputs and
filler outputs are discarded. Floating-point results can vary with physical
batch shape and numerical backend; use consistent settings when comparing scores.

## Custom scorer

```python
import math
from netab.scorers.base import Scorer


class GeometricAmountScorer(Scorer):
    def score(self, candidate, history, label):
        amount = int(candidate[0])
        return amount * math.log(2)
```

This illustrative independent score cancels out of relative selection because
it does not depend on history. A learned conditional scorer can instead inspect
the complete prefix and label. Scorers must be deterministic for fixed inputs
during generation; put neural models in evaluation mode.

Subclassing `Scorer` supplies a default `score_many(queries)` loop. Override it
to batch work while preserving query order. Each query is a
`(candidate, history, label)` tuple and must yield one finite scalar. A simple
object implementing only `score` is also accepted by the generator.

## Training

`fit_scorer` trains a Tiny or Hugging Face scorer on its selected device using next-token loss on
serialized active rows. Padding targets are masked out. Validation NLL is the
sum of eligible next-token losses divided by the total number of targets,
including structural tokens. This training objective differs from the
candidate-only score used during generation.

Validation data must be separate from fitting data in an application. The
lowest validation NLL selects the restored state; ties retain the earlier
epoch. The optional saved checkpoint contains the selected weights and
validation metadata. No dataset is serialized into it.

## Troubleshooting

### Why does a tokenizer reject a value boundary?

A model token can sometimes cover characters from adjacent pieces. The wrapper
rejects ambiguous value spans instead of scoring an incorrect token subset.
Enable tokenizer extensions or use a tokenizer that keeps the spans separate.
A multi-token value is allowed if every token stays inside its value span.

### What if a row is too long for a causal model?

The wrapper raises an error rather than truncating history. Choose a larger
context model, reduce schema capacity where appropriate, or extend value tokens
to reduce fragmentation. Do not silently drop active records to fit a context.

### How do I recover a trained scorer?

The CLI accepts its training directory through `--checkpoint`. Tiny also restores
its registered class order and architecture. Use the same schema and quantizer.
Hugging Face generation loads the saved local model and tokenizer. Markov fits
from the supplied fitting rows rather than loading neural weights.
