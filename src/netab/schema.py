"""Configurable record domains and prefix-padded rows."""

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence, Tuple
import math

Record = Tuple[Any, ...]


@dataclass(frozen=True)
class Field:
    name: str
    dtype: str = "float"
    minimum: float = None
    maximum: float = None
    choices: tuple = ()

    def __post_init__(self):
        if not self.name or self.dtype not in ("float", "int", "str"):
            raise ValueError("field needs a name and a supported dtype")
        if self.minimum is not None and self.maximum is not None and self.minimum > self.maximum:
            raise ValueError("minimum exceeds maximum")

    def validate(self, value):
        if self.dtype not in ("float", "int", "str"):
            raise ValueError("dtype must be float, int or str")
        if self.dtype == "str":
            if not isinstance(value, str):
                raise ValueError(f"{self.name}: expected string")
            x = value
        else:
            x = float(value)
            if not math.isfinite(x) or (self.dtype == "int" and not x.is_integer()):
                raise ValueError(f"{self.name}: invalid numeric value")
            if self.minimum is not None and x < self.minimum:
                raise ValueError(f"{self.name}: below minimum")
            if self.maximum is not None and x > self.maximum:
                raise ValueError(f"{self.name}: above maximum")
        if self.choices and x not in self.choices:
            raise ValueError(f"{self.name}: outside choices")


@dataclass(frozen=True)
class Schema:
    positions: int
    fields: Sequence[Field]
    null_record: Record
    activity_field: str = None
    inactive_value: Any = 0
    activity: Callable = None
    record_validator: Callable = None

    def __post_init__(self):
        if type(self.positions) is not int or self.positions < 1 or not self.fields:
            raise ValueError("positive positions and nonempty fields required")
        if len(set(f.name for f in self.fields)) != len(self.fields):
            raise ValueError("duplicate field names")
        if len(self.null_record) != len(self.fields):
            raise ValueError("null record width differs from schema")
        if self.activity_field is not None and self.activity_field not in self.names:
            raise ValueError("unknown activity field")
        if self.active(self.null_record):
            raise ValueError("null record must be inactive")

    @property
    def names(self):
        return tuple(f.name for f in self.fields)

    def active(self, record):
        if self.activity is not None:
            return bool(self.activity(tuple(record)))
        if self.activity_field is None:
            return any(
                v != z if f.dtype == "str" else float(v) != float(z)
                for f, v, z in zip(self.fields, record, self.null_record)
            )
        i = self.names.index(self.activity_field)
        if self.fields[i].dtype == "str":
            return record[i] != self.inactive_value
        return float(record[i]) != float(self.inactive_value)

    def validate_record(self, record):
        if len(record) != len(self.fields) or not self.active(record):
            raise ValueError("invalid active record shape or activity")
        for f, value in zip(self.fields, record):
            f.validate(value)
        if self.record_validator and not self.record_validator(tuple(record)):
            raise ValueError("record predicate failed")

    def length(self, row):
        if len(row) != self.positions:
            raise ValueError("row must contain exactly P records")
        length = 0
        padding = False
        for record in row:
            if len(record) != len(self.fields):
                raise ValueError("incorrect record width")
            if self.active(record):
                if padding:
                    raise ValueError("active record after padding (V1)")
                self.validate_record(record)
                length += 1
            else:
                padding = True
                for f, v, z in zip(self.fields, record, self.null_record):
                    equal = v == z if f.dtype == "str" else float(v) == float(z)
                    if not equal:
                        raise ValueError("padding must equal null record (V1)")
        return length

    def pad(self, records):
        if len(records) > self.positions:
            raise ValueError("too many records")
        row = [tuple(r) for r in records]
        row += [tuple(self.null_record)] * (self.positions - len(row))
        self.length(row)
        return row

    @classmethod
    def from_dict(cls, config: Mapping):
        cfg = dict(config)
        cfg["fields"] = [Field(**f) for f in cfg["fields"]]
        cfg["null_record"] = tuple(cfg["null_record"])
        return cls(**cfg)
