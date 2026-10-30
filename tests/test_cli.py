import csv
import json
import yaml
from netab.cli import main
from netab.io import read_csv, write_csv
from netab import Schema


def test_tiny_training_and_generation_cli(tmp_path, capsys):
    config = {
        "schema": {
            "positions": 2,
            "fields": [{"name": "amount", "dtype": "int", "minimum": 1}],
            "null_record": [0],
        },
        "scorers": {"tiny": {"d_model": 16, "heads": 2, "layers": 1, "dropout": 0}},
        "training": {"epochs": 1, "batch_size": 2, "num_threads": 1},
        "generation": {"k": 2, "m": 2},
    }
    config_path = tmp_path / "custom.yaml"
    config_path.write_text(yaml.safe_dump(config))
    schema = Schema.from_dict(config["schema"])
    fitting = tmp_path / "fitting.csv"
    validation = tmp_path / "validation.csv"
    write_csv(fitting, [schema.pad([(1,), (2,)]), schema.pad([(3,)])], ["A", "A"], schema, {})
    write_csv(validation, [schema.pad([(4,)])], ["A"], schema, {})
    checkpoint = tmp_path / "model"
    main(
        [
            "train",
            "--config",
            str(config_path),
            "--data",
            str(fitting),
            "--validation-data",
            str(validation),
            "--threads",
            "2",
            "--scorer",
            "tiny",
            "--output",
            str(checkpoint),
        ]
    )
    assert (checkpoint / "selected.pt").is_file()
    output = tmp_path / "generated.csv"
    main(
        [
            "generate",
            "--config",
            str(config_path),
            "--data",
            str(fitting),
            "--threads",
            "2",
            "--scorer",
            "tiny",
            "--checkpoint",
            str(checkpoint),
            "--class-label",
            "A",
            "--rows",
            "3",
            "--output",
            str(output),
        ]
    )
    actual, labels = read_csv(output, schema, {})
    assert len(actual) == 3 and labels == ["A"] * 3
    metadata = json.loads((checkpoint / "validation.json").read_text())
    assert metadata["selected_epoch"] == 1


def test_markov_uniform_and_validation_cli(tmp_path):
    config = tmp_path / "config.yaml"
    config.write_text("schema:\n  positions: 1\n  fields: [{name: amount, dtype: int}]\n  null_record: [0]\n")
    data = tmp_path / "input.csv"
    with data.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["amount_0", "class"])
        writer.writerows([["2", "A"], ["3", "A"]])
    main(["validate", "--config", str(config), "--data", str(data)])
    for rule in ["relative", "uniform", "absolute"]:
        main(
            [
                "generate",
                "--config",
                str(config),
                "--data",
                str(data),
                "--class-label",
                "A",
                "--rows",
                "3",
                "--rule",
                rule,
                "--output",
                str(tmp_path / f"{rule}.csv"),
            ]
        )
