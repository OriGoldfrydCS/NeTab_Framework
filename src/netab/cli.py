"""Command-line entry points. Neural computation supports automatic device selection."""

import argparse
import json
from pathlib import Path
from . import Field, Schema, Quantizer, Serializer, EmpiricalProposal, Generator, MarkovScorer
from .io import load_config, read_csv, write_csv


def toy():
    schema = Schema(
        4, [Field("amount", "int", minimum=1), Field("sign", "int", choices=(-1, 1))], (0, 0), "sign"
    )
    rows = [schema.pad([(i + 1, 1), (i + 2, -1)][: 1 + i % 2]) for i in range(12)]
    return schema, rows, ["demo"] * len(rows)


def make_scorer(kind, schema, quantizer, labels, options, device="auto"):
    serializer = Serializer(schema, quantizer)
    if kind == "markov":
        return MarkovScorer(quantizer, **options.get("markov", {}))
    if kind == "tiny":
        from .scorers.tiny import TinyTransformerScorer

        return TinyTransformerScorer(
            serializer, sorted(set(labels)), **{**options.get("tiny", {}), "device": device}
        )
    from .scorers.hf import HFCausalLMScorer

    hf = dict(options.get("hf", {}))
    name = hf.pop("model", "gpt2")
    hf["device"] = device
    return HFCausalLMScorer.from_pretrained(name, serializer, **hf)


def main(argv=None):
    parser = argparse.ArgumentParser(prog="netab", description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    quick = sub.add_parser("quickstart", help="generate from an in-memory toy dataset")
    quick.add_argument("--seed", type=int, default=42)
    quick.add_argument("--rows", type=int, default=16)
    for command in ("validate", "generate", "train"):
        p = sub.add_parser(command)
        p.add_argument("--config", required=True)
        p.add_argument("--data", required=True)
        if command != "validate":
            p.add_argument("--scorer", choices=["markov", "tiny", "hf"], default="markov")
            p.add_argument("--output", required=True)
            p.add_argument("--device", choices=["auto", "cpu", "cuda"])
            p.add_argument(
                "--threads", type=int, help="optional PyTorch thread count; default leaves it unchanged"
            )
        if command == "train":
            p.add_argument("--validation-data", required=True)
        if command == "generate":
            p.add_argument("--class-label", required=True)
            p.add_argument("--rows", type=int, default=1000)
            p.add_argument("--seed", type=int, default=42)
            p.add_argument("--class-index", type=int, default=0)
            p.add_argument("--rng-namespace", type=int, default=1)
            p.add_argument("--rule", choices=["relative", "uniform", "absolute"])
            p.add_argument("--checkpoint")
    args = parser.parse_args(argv)
    if args.command == "quickstart":
        schema, rows, labels = toy()
        quantizer = Quantizer(schema)
        scorer = MarkovScorer(quantizer).fit(rows, labels, schema)
        result = Generator(EmpiricalProposal(schema, rows, labels), scorer).generate(
            "demo", args.rows, args.seed
        )
        print(
            json.dumps({"rows": len(result.rows), "valid": all(schema.length(r) >= 0 for r in result.rows)})
        )
        return
    cfg, schema, quantizer = load_config(args.config)
    rows, labels = read_csv(args.data, schema, cfg.get("data", {}))
    if args.command == "validate":
        print(json.dumps({"rows": len(rows), "valid": True}))
        return
    device = args.device or cfg.get("device", "auto")
    if args.threads is not None and args.threads < 1:
        parser.error("--threads must be a positive integer")
    if args.command == "train":
        if args.scorer == "markov":
            parser.error("Markov is fitted directly during generate; train supports tiny or hf")
        import torch
        from .training import fit_scorer

        options = dict(cfg.get("training", {}))
        options["device"] = device
        if args.threads is not None:
            options["num_threads"] = args.threads
        num_threads = options.get("num_threads")
        if num_threads is not None:
            if type(num_threads) is not int or num_threads < 1:
                parser.error("training.num_threads must be a positive integer or null")
            torch.set_num_threads(num_threads)
        torch.manual_seed(options.get("seed", 42))
        scorer = make_scorer(args.scorer, schema, quantizer, labels, cfg.get("scorers", {}), device=device)
        output = Path(args.output)
        output.mkdir(parents=True, exist_ok=False)
        validation_rows, validation_labels = read_csv(args.validation_data, schema, cfg.get("data", {}))
        result = fit_scorer(
            scorer, rows, labels, validation_rows, validation_labels, output=output / "selected.pt", **options
        )
        if args.scorer == "hf":
            scorer.model.save_pretrained(output / "model")
            scorer.tokenizer.save_pretrained(output / "model")
        if args.scorer == "tiny":
            metadata = {"classes": sorted(set(labels)), "options": cfg.get("scorers", {}).get("tiny", {})}
            (output / "scorer.json").write_text(json.dumps(metadata, indent=2))
        (output / "validation.json").write_text(json.dumps(result, indent=2))
        print(json.dumps(result))
        return
    options = dict(cfg.get("generation", {}))
    if args.rule:
        options["rule"] = args.rule
    scorer = None
    if args.threads is not None:
        import torch

        torch.set_num_threads(args.threads)
    if options.get("rule", "relative") != "uniform":
        if args.scorer in ("tiny", "hf") and not args.checkpoint:
            parser.error("neural generation requires --checkpoint pointing to its training directory")
        if args.scorer == "hf" and args.checkpoint:
            from .scorers.hf import HFCausalLMScorer

            scorer = HFCausalLMScorer.from_pretrained(
                str(Path(args.checkpoint) / "model"),
                Serializer(schema, quantizer),
                extend_tokens=False,
                local_files_only=True,
                device=device,
                **{
                    key: value
                    for key, value in cfg.get("scorers", {}).get("hf", {}).items()
                    if key in ("normalize_per_value", "max_length", "scoring_batch_size", "fixed_length")
                },
            )
        elif args.scorer == "tiny":
            from .scorers.tiny import TinyTransformerScorer

            metadata = json.loads((Path(args.checkpoint) / "scorer.json").read_text())
            scorer = TinyTransformerScorer(
                Serializer(schema, quantizer),
                metadata["classes"],
                **{**metadata["options"], "device": device},
            )
        else:
            scorer = make_scorer(
                args.scorer, schema, quantizer, labels, cfg.get("scorers", {}), device=device
            )
        if args.scorer == "markov":
            scorer.fit(rows, labels, schema)
        elif args.scorer == "tiny":
            if not args.checkpoint:
                parser.error("tiny generation requires --checkpoint pointing to its training directory")
            import torch

            state = torch.load(Path(args.checkpoint) / "selected.pt", map_location="cpu", weights_only=True)
            scorer.model.load_state_dict(state["model_state_dict"])
        elif not args.checkpoint:
            parser.error("hf generation requires --checkpoint pointing to its training directory")
    proposal = EmpiricalProposal(schema, rows, labels)
    result = Generator(proposal, scorer, **options).generate(
        args.class_label, args.rows, args.seed, dataset_id=args.rng_namespace, class_index=args.class_index
    )
    write_csv(args.output, result.rows, [args.class_label] * args.rows, schema, cfg.get("data", {}))
    print(json.dumps({"rows": len(result.rows), "valid": True}))


if __name__ == "__main__":
    main()
