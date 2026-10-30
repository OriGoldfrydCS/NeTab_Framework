import torch
from netab import Serializer, Quantizer
from netab.cli import toy
from netab.scorers.tiny import TinyTransformerScorer
from netab.training import fit_scorer


def test_tiny_causality_and_validation_selection(tmp_path):
    torch.set_num_threads(2)
    torch.manual_seed(7)
    schema, rows, labels = toy()
    scorer = TinyTransformerScorer(
        Serializer(schema, Quantizer(schema)), ["demo"], d_model=16, heads=2, dropout=0
    )
    encoded = scorer.encode("demo", rows[1][:2])
    x = torch.tensor([encoded.ids])
    y = x.clone()
    y[0, -1] = (y[0, -1] + 1) % len(scorer.vocabulary)
    scorer.model.eval()
    with torch.inference_mode():
        assert torch.equal(scorer.model(x).logits[:, :-1], scorer.model(y).logits[:, :-1])
    metadata = fit_scorer(
        scorer,
        rows[:8],
        labels[:8],
        rows[8:],
        labels[8:],
        epochs=2,
        batch_size=4,
        learning_rate=0.001,
        num_threads=2,
        output=tmp_path / "selected.pt",
    )
    assert metadata["validation_nll"] == min(e["validation_nll"] for e in metadata["history"])
    assert torch.isfinite(torch.tensor(scorer.score(rows[0][0], [], "demo")))


def test_device_auto_resolution_without_allocating_on_gpu(monkeypatch):
    from netab.devices import resolve_device

    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    assert resolve_device("auto") == torch.device("cpu")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    assert resolve_device("auto") == torch.device("cuda")
    assert resolve_device("cpu") == torch.device("cpu")
    assert resolve_device("cuda:1") == torch.device("cuda:1")


def test_training_threads_are_opt_in_and_have_no_fixed_cap(monkeypatch):
    schema, rows, labels = toy()
    scorer = TinyTransformerScorer(
        Serializer(schema, Quantizer(schema)),
        ["demo"],
        layers=1,
        d_model=16,
        heads=2,
        dropout=0,
        device="cpu",
    )
    requested = []
    # Record requests without changing actual test resources.
    monkeypatch.setattr(torch, "set_num_threads", requested.append)
    fit_scorer(scorer, rows[:2], labels[:2], rows[2:3], labels[2:3], epochs=1, device="cpu")
    assert requested == []
    fit_scorer(scorer, rows[:2], labels[:2], rows[2:3], labels[2:3], epochs=1, device="cpu", num_threads=12)
    assert requested == [12]
