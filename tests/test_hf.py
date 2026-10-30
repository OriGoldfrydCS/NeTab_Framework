"""Offline token-span and likelihood checks using a locally constructed toy LM."""

import numpy as np
import pytest
import torch

from netab import Field, Schema, Serializer, Quantizer
from netab.scorers.hf import HFCausalLMScorer

transformers = pytest.importorskip("transformers")
tokenizers = pytest.importorskip("tokenizers")


def toy_hf():
    torch.set_num_threads(2)
    torch.manual_seed(2)
    schema = Schema(2, [Field("a", "int"), Field("b", "int")], (0, 0))
    serializer = Serializer(schema, Quantizer(schema))
    # Character tokens deliberately fragment quantized value symbols.
    text = serializer.text("A", [(1, 65536)])
    alphabet = sorted(set(text))
    vocab = {v: i for i, v in enumerate(["<UNK>", "<PAD>", "<EOS>"] + alphabet)}
    backend = tokenizers.Tokenizer(tokenizers.models.WordLevel(vocab, unk_token="<UNK>"))
    backend.pre_tokenizer = tokenizers.pre_tokenizers.Split("", behavior="isolated")
    tokenizer = transformers.PreTrainedTokenizerFast(
        tokenizer_object=backend, unk_token="<UNK>", pad_token="<PAD>", eos_token="<EOS>"
    )
    conf = transformers.GPT2Config(
        vocab_size=len(tokenizer),
        n_positions=256,
        n_embd=16,
        n_layer=1,
        n_head=2,
        resid_pdrop=0,
        embd_pdrop=0,
        attn_pdrop=0,
    )
    conf._attn_implementation = "eager"
    model = transformers.GPT2LMHeadModel(conf)
    scorer = HFCausalLMScorer(model, tokenizer, serializer, max_length=256, scoring_batch_size=2)
    return scorer


def test_multi_token_value_loss_and_normalization():
    scorer = toy_hf()
    candidate = (1, 65536)
    encoded = scorer.encode("A", [candidate])
    fields = encoded.values[0]
    assert len(fields[0]) > 1 and len(fields[0]) != len(fields[1])
    ids = torch.tensor([encoded.ids])
    scorer.model.eval()
    losses = []
    with torch.inference_mode():
        logits = scorer.model(ids, use_cache=False).logits[0].double()
        for field in sorted(fields):
            ix = torch.tensor(fields[field])
            losses.append(-logits[ix - 1].log_softmax(-1).gather(-1, ids[0, ix, None]).squeeze(-1))
    expected = float(torch.cat(losses).mean())
    assert scorer.score(candidate, [], "A") == pytest.approx(expected, abs=1e-7)
    scorer.normalize_per_value = True
    per_value = float(torch.stack([loss.mean() for loss in losses]).mean())
    assert scorer.score(candidate, [], "A") == pytest.approx(per_value, abs=1e-7)
    assert abs(per_value - expected) > 1e-8


def test_hf_padding_batching_and_context_errors():
    scorer = toy_hf()
    queries = [((1, 65536), [], "A"), ((2, 10), [], "A"), ((3, 2), [], "A")]
    together = scorer.score_many(queries)
    separately = np.array([scorer.score(*q) for q in queries])
    assert np.allclose(together, separately, atol=1e-6, rtol=0)
    scorer.fixed_length = 256
    assert np.allclose(together, scorer.score_many(queries), atol=1e-6, rtol=0)
    scorer.max_length = 5
    with pytest.raises(ValueError, match="context"):
        scorer.encode("A", [(1, 2)])


def test_local_hf_cli_training_save_and_reload(tmp_path):
    import yaml
    from netab.cli import main
    from netab.io import write_csv, read_csv

    scorer = toy_hf()
    base = tmp_path / "base"
    scorer.model.save_pretrained(base)
    scorer.tokenizer.save_pretrained(base)
    config = {
        "schema": {
            "positions": 2,
            "fields": [{"name": "a", "dtype": "int"}, {"name": "b", "dtype": "int"}],
            "null_record": [0, 0],
        },
        "scorers": {"hf": {"model": str(base), "local_files_only": True, "max_length": 256}},
        "training": {"epochs": 1, "num_threads": 2, "batch_size": 2},
        "generation": {"k": 2, "m": 2},
    }
    path = tmp_path / "hf.yaml"
    path.write_text(yaml.safe_dump(config))
    schema = Schema.from_dict(config["schema"])
    fitting, validation = tmp_path / "fit.csv", tmp_path / "val.csv"
    write_csv(fitting, [schema.pad([(1, 2), (3, 4)]), schema.pad([(2, 3)])], ["A", "A"], schema, {})
    write_csv(validation, [schema.pad([(5, 6)])], ["A"], schema, {})
    checkpoint = tmp_path / "trained"
    main(
        [
            "train",
            "--config",
            str(path),
            "--data",
            str(fitting),
            "--validation-data",
            str(validation),
            "--scorer",
            "hf",
            "--output",
            str(checkpoint),
        ]
    )
    assert (checkpoint / "model/config.json").is_file()
    output = tmp_path / "generated.csv"
    main(
        [
            "generate",
            "--config",
            str(path),
            "--data",
            str(fitting),
            "--scorer",
            "hf",
            "--checkpoint",
            str(checkpoint),
            "--class-label",
            "A",
            "--rows",
            "2",
            "--output",
            str(output),
        ]
    )
    rows, labels = read_csv(output, schema, {})
    assert len(rows) == 2 and labels == ["A", "A"]
