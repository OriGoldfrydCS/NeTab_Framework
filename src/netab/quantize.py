"""Lossy scoring vocabulary; generation always copies the original record."""

from dataclasses import dataclass
import math


@dataclass(frozen=True)
class LogQuantizer:
    bins: int = 200
    ceiling: float = 65536

    def __post_init__(self):
        if (
            type(self.bins) is not int
            or self.bins < 1
            or self.ceiling <= 1
            or not math.isfinite(self.ceiling)
        ):
            raise ValueError("invalid quantizer parameters")

    def __call__(self, value):
        x = float(value)
        if not math.isfinite(x):
            raise ValueError("nonfinite value")
        if x < 0:
            return "<VAL_NEG>"
        if x == 0:
            return "<VAL_ZERO>"
        if x >= self.ceiling:
            return f"<VAL_{int(self.ceiling)}>"
        exponent = math.log2(self.ceiling)
        i = max(0, min(self.bins - 1, int(math.log2(max(1, x)) * self.bins / exponent)))
        return f"<VAL_{int(math.pow(2, i * exponent / self.bins))}>"

    @property
    def vocabulary(self):
        exponent = math.log2(self.ceiling)
        vals = {f"<VAL_{int(math.pow(2, i * exponent / self.bins))}>" for i in range(self.bins)}
        return sorted(vals | {"<VAL_NEG>", "<VAL_ZERO>", f"<VAL_{int(self.ceiling)}>"})


class Quantizer:
    def __init__(self, schema, per_field=None):
        self.schema = schema
        self.per_field = per_field or {f.name: LogQuantizer() for f in schema.fields}
        if set(self.per_field) != set(schema.names):
            raise ValueError("one quantizer required per field")

    def __call__(self, record):
        if len(record) != len(self.schema.fields):
            raise ValueError("incorrect record width for quantization")
        return tuple(self.per_field[f.name](v) for f, v in zip(self.schema.fields, record))

    @property
    def vocabulary(self):
        return sorted({v for q in self.per_field.values() for v in q.vocabulary})
