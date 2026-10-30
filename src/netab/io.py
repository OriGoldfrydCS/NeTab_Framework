"""CSV adapters parameterized by schema and column patterns."""

import csv
from pathlib import Path
import yaml
from .schema import Schema
from .quantize import Quantizer, LogQuantizer


def load_config(path):
    config = yaml.safe_load(Path(path).read_text())
    schema = Schema.from_dict(config["schema"])
    options = config.get("quantizer", {})
    per_field = {name: LogQuantizer(**options.get(name, options.get("default", {}))) for name in schema.names}
    return config, schema, Quantizer(schema, per_field)


def columns(schema, config):
    patterns = config.get("columns", {name: name + "_{j}" for name in schema.names})
    groups = [[patterns[name].format(j=j) for j in range(schema.positions)] for name in schema.names]
    flattened = [c for group in groups for c in group]
    if len(set(flattened)) != len(flattened) or config.get("class_column", "class") in flattened:
        raise ValueError("CSV column patterns must produce unique columns distinct from the class column")
    return groups


def read_csv(path, schema, config):
    groups = columns(schema, config)
    class_column = config.get("class_column", "class")
    rows, labels = [], []
    with Path(path).open(newline="") as f:
        reader = csv.DictReader(f)
        required = {class_column} | {c for group in groups for c in group}
        if len(set(reader.fieldnames or [])) != len(reader.fieldnames or []):
            raise ValueError("duplicate CSV columns")
        if not required.issubset(reader.fieldnames or []):
            raise ValueError("CSV is missing schema or class columns")
        for raw in reader:
            row = [tuple(raw[group[j]] for group in groups) for j in range(schema.positions)]
            schema.length(row)
            rows.append(row)
            labels.append(raw[class_column])
    return rows, labels


def write_csv(path, rows, labels, schema, config):
    if len(rows) != len(labels):
        raise ValueError("rows and labels differ in length")
    groups = columns(schema, config)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([c for group in groups for c in group] + [config.get("class_column", "class")])
        for row, label in zip(rows, labels):
            schema.length(row)
            writer.writerow(
                [row[j][g] for g in range(len(groups)) for j in range(schema.positions)] + [label]
            )
