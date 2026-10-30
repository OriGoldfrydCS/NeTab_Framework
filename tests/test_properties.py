from hypothesis import given, settings, strategies as st
from netab import Schema, Field, EmpiricalProposal, Generator, validate_rows


@settings(max_examples=20, deadline=None)
@given(p=st.integers(1, 8), f=st.integers(1, 5), seed=st.integers(0, 10000))
def test_schema_independent_v1_v2(p, f, seed):
    schema = Schema(p, [Field(f"f{i}", "int", minimum=1) for i in range(f)], (0,) * f)
    rows = [schema.pad([tuple(1 + i + j + k for k in range(f)) for j in range(1 + i % p)]) for i in range(8)]
    result = Generator(EmpiricalProposal(schema, rows, ["A"] * 8), rule="uniform", k=3, m=2).generate(
        "A", 8, seed
    )
    assert validate_rows(schema, result.rows)
    for row in result.rows:
        length = schema.length(row)
        assert all(r == schema.null_record for r in row[length:])
