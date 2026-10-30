# Configuration

`configs/example_custom.yaml` is a generic starting point.
`configs/cesnet_quic22.yaml` is a traffic usage example. Parameter values below
are software defaults and architectural settings, not performance claims.

## Schema and CSV fields

| Setting | Default | Meaning |
| --- | --- | --- |
| `schema.positions` | Required | Maximum number of records, `P` |
| `schema.fields` | Required | Ordered field definitions, determining `F` |
| `fields[].name` | Required | Unique field name |
| `fields[].dtype` | `float` | `int`, `float`, or `str` |
| `fields[].minimum`, `maximum` | None | Inclusive active-value bounds |
| `fields[].choices` | Empty | Optional permitted active values |
| `schema.null_record` | Required | Exact trailing padding record |
| `schema.activity_field` | None | Field compared with inactive value; otherwise compare with null record |
| `schema.inactive_value` | `0` | Inactive marker for activity field |
| `data.class_column` | `class` | CSV class label column |
| `data.columns` | `<name>_{j}` | Per-field column patterns, zero-based position |

Python also accepts activity and record-validator callables; see
[data format](data_format.md). Numeric strings in CSV are validated numerically
and preserved verbatim for output.

## Quantizer

```yaml
quantizer:
  default: {bins: 200, ceiling: 65536}
  amount: {bins: 100, ceiling: 1024}
```

A field-specific mapping overrides the default mapping for that field. The
logarithmic quantizer maps negative values to `NEG`, exact zero to `ZERO`, and
values at or above `ceiling` to the ceiling symbol. For other positive values:

```text
i = clamp(floor(log2(max(1, x)) * bins / log2(ceiling)), 0, bins - 1)
symbol = floor(2 ** (i * log2(ceiling) / bins))
```

Symbols use `<VAL_...>` spelling. Integer flooring can make different bins
share a symbol. Values between zero and one map like one. Nonfinite values
are rejected. Each field can use a different quantizer through the Python API.
Changing quantizers changes the scoring vocabulary and requires compatible
neural weights; it does not alter copied raw values.

## Generation

| Setting | Default | Guidance |
| --- | --- | --- |
| `generation.k` | `8` | Candidate slots per position; larger values increase scorer work |
| `generation.m` | `4` | Reference histories per class/seed/length; larger values increase scorer work |
| `generation.temperature` | `1.0` | Positive finite softmax temperature; smaller values concentrate mass |
| `generation.rule` | `relative` | `relative`, `uniform`, or `absolute` |

`K` and `M` must be positive integers. `uniform` needs no scorer. `absolute`
ignores reference scores; `relative` compares current and reference histories.
A temperature has no effect on uniform selection.

`Generator.generate` takes a class, output count, and seed (default 42). It also
accepts a class index and dataset namespace, both nonnegative integers, to
separate random streams. The namespace is only a random-stream identifier;
it does not select data. Identical input ordering and settings reproduce the
same proposal draws. An optional in-memory `ProposalPlan` supports replay and
is validated for shapes, fitting-pool eligibility, empirical lengths, and
uniform ranges before use.

A fixed seed alone does not guarantee identical neural probabilities across
numerical backends or versions. Raw records and candidate order also depend on
the order of fitting rows.

## Scorers

| Setting | Default | Meaning |
| --- | --- | --- |
| `scorers.markov.smoothing` | `0.5` | Additive base-count smoothing |
| `scorers.markov.backoff` | `10.0` | Strength of interpolation toward base probabilities |
| `scorers.markov.vocabulary_size` | Quantizer union plus unknown | Optional explicit smoothing vocabulary size |
| `scorers.tiny.layers` | `2` | Causal Transformer layers |
| `scorers.tiny.d_model` | `256` | Embedding width |
| `scorers.tiny.heads` | `4` | Attention heads; width must divide evenly |
| `scorers.tiny.dropout` | `0.1` | Training dropout |
| `scorers.hf.model` | `gpt2` | Hugging Face name or local directory |
| `scorers.hf.from_scratch` | `false` | Random model initialization |
| `scorers.hf.extend_tokens` | `true` | Add delimiter and quantizer symbols |
| `scorers.hf.local_files_only` | `false` | Require cached/local model files |
| `scorers.hf.normalize_per_value` | `false` | Average token NLL within each field, then across fields |
| `scorers.hf.max_length` | Model context capacity | Maximum token count without truncation |
| `scorers.hf.scoring_batch_size` | `4` | Physical scoring batch size |
| `scorers.hf.fixed_length` | None | Optional fixed right-padding width |

The traffic example sets the Hugging Face context limit to 512. Parameter
counts depend on vocabulary, context length, and model architecture; inspect
the instantiated model rather than assuming a fixed size.

## Training

| `training` setting | Default | Meaning |
| --- | --- | --- |
| `epochs` | `15` | Number of complete fitting passes |
| `batch_size` | `4` | Rows per optimization batch |
| `learning_rate` | `0.00003` | AdamW learning rate |
| `weight_decay` | `0.01` | AdamW weight decay |
| `max_grad_norm` | `1.0` | Gradient clipping threshold |
| `seed` | `42` | Training order/dropout seed |
| `num_threads` | `null` / `None` | Optional positive PyTorch thread count; no fixed upper limit |

Set the initialization seed before creating a neural scorer when using the
Python API. The CLI does this automatically. `fit_scorer` resets random state
for training but cannot undo initialization that already occurred.

Use separate fitting and validation inputs. The selected model minimizes
validation token NLL; validation data is never added to donor pools. The CLI
stores selected neural weights under its output directory and refuses to
reuse an existing training directory. Generation similarly refuses to
overwrite an existing CSV.

## Device selection and candidate filtering

Top-level YAML `device` defaults to `auto`: choose CUDA if available, otherwise
CPU. CLI `--device auto|cpu|cuda` overrides YAML. The Python neural scorer
constructors and `fit_scorer` also accept `device="auto"` or an explicit device,
including an indexed CUDA device such as `cuda:0`. Markov remains on CPU.

`Generator(candidate_filter=...)` accepts a Python callable taking raw history
and a candidate record. It is not evaluated from YAML. Rejected sampled slots
are excluded before scoring, their probabilities are zero, and all-rejected
steps raise an error without fallback. See [FAQ](faq.md) for a complete example.
