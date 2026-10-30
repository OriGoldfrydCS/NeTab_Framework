"""A different record width and activity predicate."""

from netab import Field, Schema, EmpiricalProposal, Generator, validate_rows

schema = Schema(
    5,
    [Field("volume", "int", minimum=1)],
    (0,),
    activity=lambda r: r[0] != 0,
    record_validator=lambda r: r[0] % 2 == 0,
)
rows = [schema.pad([(2,), (4,)]), schema.pad([(6,)])]
result = Generator(EmpiricalProposal(schema, rows, ["A", "A"]), rule="uniform", k=3, m=2).generate("A", 8)
assert validate_rows(schema, result.rows)
print("Custom schema passed V1 and V2.")
