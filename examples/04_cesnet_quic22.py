"""Use a local adapted CESNET CSV; optionally open the official download page."""

import argparse
import webbrowser
from pathlib import Path
from netab.cli import main

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--data", type=Path)
parser.add_argument("--class-label", default="Youtube")
parser.add_argument("--output", default="outputs/cesnet.csv")
parser.add_argument("--download", action="store_true", help="open the official dataset download record")
args = parser.parse_args()
if args.download:
    webbrowser.open("https://zenodo.org/records/10728760")
if args.data and args.data.is_file():
    config = Path(__file__).resolve().parents[1] / "configs/cesnet_quic22.yaml"
    main(
        [
            "generate",
            "--config",
            str(config),
            "--data",
            str(args.data),
            "--class-label",
            args.class_label,
            "--rows",
            "1000",
            "--output",
            args.output,
        ]
    )
else:
    print("Obtain CESNET-QUIC22 from its official record, adapt its PPI columns, and supply --data.")
