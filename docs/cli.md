# CLI

After an editable installation, `netab` is available as a console command.
`python -m netab.cli` invokes the same entry point. Run `netab --help` or
`netab <command> --help` for argument descriptions. Neural commands default to automatic CUDA/CPU selection; use `--device cpu`
to require CPU.
All paths below are relative example paths provided by the caller.

## Quickstart

```bash
netab quickstart --rows 16 --seed 42
```

Creates toy rows in memory, fits Markov, generates rows, and prints a JSON
summary. No input files or output tables are required. `--rows` defaults to
16 and `--seed` to 42.

## Validate a CSV

```bash
netab validate --config configs/example_custom.yaml --data data/fitting.csv
```

Checks all rows against the configured schema and reports the row count.
The CSV must use the configured column patterns and class column. Both
`--config` and `--data` are required.

## Generate with Markov

```bash
netab generate --config configs/example_custom.yaml \
  --data data/fitting.csv --scorer markov --class-label A \
  --rows 100 --seed 42 --output outputs/generated.csv
```

Fits Markov on the supplied data, builds donor pools from those same fitting
rows, and generates the requested class. Class labels are matched as CSV
strings. A missing class is an error.

Required generation arguments are `--config`, `--data`, `--class-label`, and
`--output`. Optional arguments:

| Argument | Default | Meaning |
| --- | --- | --- |
| `--threads` | Unset | Optional positive PyTorch thread count; otherwise preserve its current setting |
| `--device` | YAML value, otherwise `auto` | `auto`, `cpu`, or `cuda` |
| `--scorer` | `markov` | `markov`, `tiny`, or `hf` |
| `--rows` | `1000` | Positive output count |
| `--seed` | `42` | Proposal and selection seed |
| `--rule` | YAML value, otherwise `relative` | Override the selection rule |
| `--class-index` | `0` | Nonnegative class-specific random-stream index |
| `--rng-namespace` | `1` | Nonnegative random-stream namespace |
| `--checkpoint` | None | Training output directory for a neural scorer |

Uniform generation needs no fitted scorer or neural checkpoint:

```bash
netab generate --config configs/example_custom.yaml \
  --data data/fitting.csv --class-label A --rule uniform \
  --rows 100 --output outputs/uniform.csv
```

## Train Tiny

```bash
netab train --config configs/example_custom.yaml \
  --data data/fitting.csv --validation-data data/validation.csv \
  --scorer tiny --output checkpoints/tiny
```

The command reads training options from YAML, initializes the model, trains,
and restores the selected validation checkpoint. The new output directory
contains `selected.pt`, `scorer.json`, and `validation.json`. It is created
exclusively to protect existing outputs.

```bash
netab generate --config configs/example_custom.yaml \
  --data data/fitting.csv --class-label A --scorer tiny \
  --checkpoint checkpoints/tiny --rows 100 --output outputs/tiny.csv
```

Use the same schema and quantizer when training and loading. The saved class
order and architecture are restored from `scorer.json`.

## Train a Hugging Face causal model

```bash
pip install -e '.[hf]'
netab train --config configs/example_custom.yaml \
  --data data/fitting.csv --validation-data data/validation.csv \
  --scorer hf --output checkpoints/causal
```

Set `scorers.hf.model` to the desired model name or local directory. Use
`scorers.hf.from_scratch: true` for random initialization. Named pretrained
models may require a network download. Training also writes a Hugging Face
model and tokenizer under the output directory's `model/` subdirectory.

```bash
netab generate --config configs/example_custom.yaml \
  --data data/fitting.csv --class-label A --scorer hf \
  --checkpoint checkpoints/causal --rows 100 --output outputs/causal.csv
```

Generation loads that local model directory. Markov does not use `train`:
it is fitted directly during generation. Training requires `--config`,
`--data`, `--validation-data`, `--scorer tiny|hf`, and `--output`.

## Traffic usage example

```bash
python examples/04_cesnet_quic22.py --download
python examples/04_cesnet_quic22.py \
  --data data/adapted_traffic.csv --class-label Youtube \
  --output outputs/traffic.csv
```

`--download` opens the official dataset record in a browser. The example
requires an existing adapted CSV to generate rows; it does not download a
large archive automatically. See [data format](data_format.md) for the expected
traffic columns and source-data attribution.

## Existing output paths

Output CSV files and training directories are created exclusively. Choose a new
output path or manage an old artifact explicitly before reusing its name.

## Thread configuration

Both `train` and `generate` accept `--threads N`. There is no fixed upper limit.
If omitted, generation leaves PyTorch's thread setting unchanged. Training uses
`training.num_threads` from YAML if supplied; its default is `null`, which also
leaves PyTorch's setting unchanged. The Python trainer exposes the equivalent
`num_threads=None` option. An explicit CLI value overrides the training YAML.

```bash
netab train --config configs/example_custom.yaml \
  --data data/fitting.csv --validation-data data/validation.csv \
  --scorer tiny --device cpu --threads 12 --output checkpoints/tiny
```
