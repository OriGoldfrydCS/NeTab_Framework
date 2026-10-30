"""Next-token training and checkpoint selection by validation token NLL."""

from pathlib import Path
import copy
import random
import numpy as np
import torch
from torch.nn import functional as F
from .devices import resolve_device


def fit_scorer(
    scorer,
    train_rows,
    train_labels,
    validation_rows,
    validation_labels,
    epochs=15,
    batch_size=4,
    learning_rate=3e-5,
    weight_decay=0.01,
    max_grad_norm=1.0,
    seed=42,
    num_threads=None,
    output=None,
    device="auto",
):
    """Fit a scorer; num_threads=None preserves PyTorch's current thread setting."""
    if epochs < 1 or batch_size < 1:
        raise ValueError("epochs and batch size must be positive")
    if num_threads is not None:
        if type(num_threads) is not int or num_threads < 1:
            raise ValueError("num_threads must be a positive integer or None")
        torch.set_num_threads(num_threads)
    if len(train_rows) != len(train_labels) or len(validation_rows) != len(validation_labels):
        raise ValueError("labels and rows differ in length")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    selected_device = resolve_device(device)
    scorer.device = selected_device
    schema = scorer.serializer.schema

    def encode(rows, labels):
        return [scorer.encode(a, r[: schema.length(r)]).ids for r, a in zip(rows, labels)]

    train, validation = encode(train_rows, train_labels), encode(validation_rows, validation_labels)
    if not train or not validation or any(len(x) < 2 for x in train + validation):
        raise ValueError("nonempty training and validation rows required")
    pad_id = scorer.tokenizer.pad_token_id if hasattr(scorer, "tokenizer") else 0

    def batch(sequences):
        width = max(map(len, sequences))
        ids = torch.full((len(sequences), width), pad_id, dtype=torch.long, device=selected_device)
        mask = torch.zeros_like(ids)
        for i, seq in enumerate(sequences):
            ids[i, : len(seq)] = torch.tensor(seq, device=selected_device)
            mask[i, : len(seq)] = 1
        targets = ids[:, 1:].clone()
        targets[mask[:, 1:] == 0] = -100
        return ids, mask, targets

    model = scorer.model.to(selected_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    best, best_state, history = float("inf"), None, []
    for epoch in range(1, epochs + 1):
        model.train()
        order = torch.randperm(len(train)).tolist()
        for start in range(0, len(order), batch_size):
            ids, mask, targets = batch([train[i] for i in order[start : start + batch_size]])
            logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits
            loss = F.cross_entropy(logits[:, :-1].reshape(-1, logits.shape[-1]), targets.reshape(-1))
            if not torch.isfinite(loss):
                raise ValueError("nonfinite training loss")
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
            optimizer.step()
        model.eval()
        total, count = 0.0, 0
        with torch.inference_mode():
            for start in range(0, len(validation), batch_size):
                ids, mask, targets = batch(validation[start : start + batch_size])
                logits = model(input_ids=ids, attention_mask=mask, use_cache=False).logits[:, :-1].double()
                total += float(
                    F.cross_entropy(
                        logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction="sum"
                    )
                )
                count += int((targets != -100).sum())
        nll = total / count
        history.append({"epoch": epoch, "validation_nll": nll, "validation_targets": count})
        if nll < best:
            best = nll
            best_state = copy.deepcopy(model.state_dict())
            selected_epoch = epoch
    model.load_state_dict(best_state)
    model.eval()
    result = {"selected_epoch": selected_epoch, "validation_nll": best, "history": history}
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state_dict": best_state, "metadata": result}, path)
    return result
