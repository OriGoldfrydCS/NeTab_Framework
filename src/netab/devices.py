"""Explicit or automatic CPU/CUDA selection for neural scorers."""

import torch


def resolve_device(device="auto"):
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    selected = torch.device(device)
    if selected.type not in ("cpu", "cuda"):
        raise ValueError("device must be auto, cpu or a CUDA device")
    return selected
