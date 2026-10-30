# Contributing

Keep the framework schema-driven. Changes to record width, field names, or
activity rules belong in schemas and configuration rather than scorer logic.
Document behavior changes and add a small toy-data test for meaningful fixes.

Install development dependencies and run the checks from the repository root:

```bash
pip install -e '.[dev]'
ruff check .
pytest
python -m build
```

Optional Hugging Face tests use locally constructed toy models and tokenizers;
install `.[hf,dev]` to run them. They never download model weights or data.

Use CPU for the toy checks. Do not include datasets, generated tables, model
weights, credentials, machine-specific paths, or internal reports in changes.
Keep generated artifacts under ignored working directories.

Describe the problem, resulting behavior, and validation in a pull request.
Report failures with a minimal in-memory example and package versions.
