"""Schema-driven pieces with explicit candidate value spans."""

from dataclasses import dataclass


@dataclass(frozen=True)
class Piece:
    text: str
    role: str
    position: int = -1
    field: int = -1


class Serializer:
    def __init__(self, schema, quantizer):
        self.schema, self.quantizer = schema, quantizer

    @property
    def delimiters(self):
        base = [
            "<SAMPLE_START>",
            "<APP_START>",
            "<APP_END>",
            "<PPI_START>",
            "<PPI_END>",
            "<SAMPLE_END>",
            "<SEP>",
        ]
        return base + [
            f"<PPI_{f}_{j}>" for j in range(self.schema.positions) for f in range(len(self.schema.fields))
        ]

    def pieces(self, label, records):
        if len(records) > self.schema.positions:
            raise ValueError("context exceeds schema")
        out = [
            Piece("<SAMPLE_START>", "delimiter"),
            Piece("<APP_START>", "delimiter"),
            Piece(str(label), "class"),
            Piece("<APP_END>", "delimiter"),
            Piece("<PPI_START>", "delimiter"),
        ]
        for j, record in enumerate(records):
            self.schema.validate_record(record)
            for f, value in enumerate(self.quantizer(record)):
                out.extend(
                    [
                        Piece(f"<PPI_{f}_{j}>", "delimiter", j, f),
                        Piece(value, "value", j, f),
                        Piece("<SEP>", "delimiter"),
                    ]
                )
        out.extend([Piece("<PPI_END>", "delimiter"), Piece("<SAMPLE_END>", "delimiter")])
        return out

    def text(self, label, records):
        return "".join(p.text for p in self.pieces(label, records))
