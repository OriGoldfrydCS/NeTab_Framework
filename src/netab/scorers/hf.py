"""Hugging Face causal models with exact value-token alignment."""

from dataclasses import dataclass
import numpy as np
import torch
from .base import Scorer
from ..devices import resolve_device


@dataclass
class Encoded:
    ids: list
    values: dict


class HFCausalLMScorer(Scorer):
    def __init__(
        self,
        model,
        tokenizer,
        serializer,
        normalize_per_value=False,
        max_length=None,
        scoring_batch_size=4,
        fixed_length=None,
        device="auto",
    ):
        self.device = resolve_device(device)
        self.model = model.to(self.device)
        self.tokenizer, self.serializer = tokenizer, serializer
        self.normalize_per_value = normalize_per_value
        self.max_length = max_length or getattr(model.config, "max_position_embeddings", 512)
        self.scoring_batch_size, self.fixed_length = scoring_batch_size, fixed_length
        if not tokenizer.is_fast:
            raise ValueError("a fast tokenizer with character offsets is required")
        if tokenizer.pad_token_id is None:
            raise ValueError("tokenizer needs a pad token")
        if scoring_batch_size < 1:
            raise ValueError("batch size must be positive")
        self.model.eval()

    @classmethod
    def from_pretrained(
        cls, name, serializer, from_scratch=False, extend_tokens=True, local_files_only=False, **kwargs
    ):
        from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(name, use_fast=True, local_files_only=local_files_only)
        config = AutoConfig.from_pretrained(name, local_files_only=local_files_only)
        model = (
            AutoModelForCausalLM.from_config(config)
            if from_scratch
            else AutoModelForCausalLM.from_pretrained(name, local_files_only=local_files_only)
        )
        if extend_tokens:
            tokenizer.add_special_tokens(
                {
                    "additional_special_tokens": serializer.delimiters
                    + serializer.quantizer.vocabulary
                    + ["<UNK_VAL>"]
                }
            )
        if tokenizer.pad_token_id is None:
            tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        model.resize_token_embeddings(len(tokenizer))
        return cls(model, tokenizer, serializer, **kwargs)

    def encode(self, label, records):
        pieces = self.serializer.pieces(label, records)
        spans, chunks, cursor = [], [], 0
        for piece in pieces:
            chunks.append(piece.text)
            spans.append((cursor, cursor + len(piece.text), piece))
            cursor += len(piece.text)
        text = "".join(chunks) + (self.tokenizer.eos_token or "")
        out = self.tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        if len(out["input_ids"]) > self.max_length:
            raise ValueError("serialized row exceeds context; truncation is forbidden")
        values = {}
        for start, end, piece in spans:
            if piece.role != "value":
                continue
            ix = [i for i, (a, b) in enumerate(out["offset_mapping"]) if a < end and b > start]
            if (
                not ix
                or ix[0] == 0
                or out["offset_mapping"][ix[0]][0] != start
                or out["offset_mapping"][ix[-1]][1] != end
                or any(
                    not start <= out["offset_mapping"][i][0] < out["offset_mapping"][i][1] <= end for i in ix
                )
            ):
                raise ValueError("token crosses a value boundary; extend tokenizer with value tokens")
            values.setdefault(piece.position, {})[piece.field] = ix
        return Encoded(out["input_ids"], values)

    def score_encoded(self, encoded):
        result = []
        self.model.eval()
        for start in range(0, len(encoded), self.scoring_batch_size):
            items = encoded[start : start + self.scoring_batch_size]
            logical = len(items)
            items = items + [items[-1]] * (self.scoring_batch_size - logical)
            width = self.fixed_length or max(len(e.ids) for e in items)
            if width > self.max_length or any(len(e.ids) > width for e in items):
                raise ValueError("invalid batch context length")
            ids = torch.full(
                (len(items), width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device
            )
            mask = torch.zeros_like(ids)
            for i, e in enumerate(items):
                ids[i, : len(e.ids)] = torch.tensor(e.ids, device=self.device)
                mask[i, : len(e.ids)] = 1
            with torch.inference_mode():
                logits = self.model(input_ids=ids, attention_mask=mask, use_cache=False).logits
                for i, e in enumerate(items[:logical]):
                    fields = e.values[max(e.values)]
                    losses = []
                    for field in sorted(fields):
                        ix = torch.tensor(fields[field], device=self.device)
                        x = logits[i, ix - 1].double()
                        target = ids[i, ix]
                        nll = torch.logsumexp(x, -1) - x.gather(-1, target[:, None]).squeeze(-1)
                        losses.append(nll.mean() if self.normalize_per_value else nll.sum())
                    denominator = (
                        len(losses) if self.normalize_per_value else sum(len(v) for v in fields.values())
                    )
                    result.append(float(torch.stack(losses).sum()) / denominator)
        return np.asarray(result)

    def score_many(self, queries):
        return self.score_encoded(
            [self.encode(label, list(history) + [candidate]) for candidate, history, label in queries]
        )

    def score(self, candidate, history, label):
        return float(self.score_many([(candidate, history, label)])[0])
