"""A small causal Transformer with a schema-derived vocabulary."""

from types import SimpleNamespace
import math
import torch
from torch import nn
from .base import Scorer
from ..devices import resolve_device


class TinyModel(nn.Module):
    def __init__(self, vocabulary_size, max_length, layers=2, d_model=256, heads=4, dropout=0.1):
        super().__init__()
        if d_model % heads:
            raise ValueError("width must be divisible by heads")
        self.config = SimpleNamespace(max_position_embeddings=max_length)
        self.embedding = nn.Embedding(vocabulary_size, d_model)
        self.position = nn.Embedding(max_length, d_model)
        block = nn.TransformerEncoderLayer(
            d_model, heads, d_model * 4, dropout, activation="gelu", batch_first=True, norm_first=True
        )
        self.blocks = nn.TransformerEncoder(block, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocabulary_size, bias=False)
        self.head.weight = self.embedding.weight

    def forward(self, input_ids, attention_mask=None, use_cache=False):
        n = input_ids.shape[1]
        if n > self.config.max_position_embeddings:
            raise ValueError("context overflow")
        x = self.embedding(input_ids) * math.sqrt(self.embedding.embedding_dim)
        x = x + self.position(torch.arange(n, device=input_ids.device))
        causal = torch.triu(torch.ones(n, n, dtype=torch.bool, device=x.device), diagonal=1)
        padding = attention_mask == 0 if attention_mask is not None else None
        x = self.blocks(x, mask=causal, src_key_padding_mask=padding)
        return SimpleNamespace(logits=self.head(self.norm(x)))


class TinyTransformerScorer(Scorer):
    def __init__(self, serializer, classes, layers=2, d_model=256, heads=4, dropout=0.1, device="auto"):
        self.device = resolve_device(device)
        self.serializer = serializer
        pieces = serializer.delimiters + serializer.quantizer.vocabulary + [str(c) for c in classes]
        self.vocabulary = list(dict.fromkeys(["<PAD>"] + pieces))
        self.index = {v: i for i, v in enumerate(self.vocabulary)}
        self.max_length = 7 + serializer.schema.positions * len(serializer.schema.fields) * 3
        self.model = TinyModel(len(self.vocabulary), self.max_length, layers, d_model, heads, dropout).to(
            self.device
        )

    def encode(self, label, records):
        from .hf import Encoded

        pieces = self.serializer.pieces(label, records)
        ids, values = [], {}
        for piece in pieces:
            if piece.text not in self.index:
                raise ValueError("unregistered class or quantized value")
            if piece.role == "value":
                values.setdefault(piece.position, {})[piece.field] = [len(ids)]
            ids.append(self.index[piece.text])
        return Encoded(ids, values)

    def score(self, candidate, history, label):
        encoded = self.encode(label, list(history) + [candidate])
        self.model.eval()
        with torch.inference_mode():
            ids = torch.tensor([encoded.ids], device=self.device)
            logits = self.model(ids).logits[0]
            ix = torch.tensor([v[0] for v in encoded.values[len(history)].values()], device=self.device)
            x = logits[ix - 1].double()
            target = ids[0, ix]
            return float((torch.logsumexp(x, -1) - x.gather(-1, target[:, None]).squeeze(-1)).mean())
