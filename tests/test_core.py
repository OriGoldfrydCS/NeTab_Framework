import math
import numpy as np
import pytest
from netab import (
    Field,
    Schema,
    LogQuantizer,
    Quantizer,
    Serializer,
    EmpiricalProposal,
    Generator,
    MarkovScorer,
    validate_rows,
)
from netab.selection import softmax, categorical
from netab.cli import toy, main
from netab.io import write_csv, read_csv


def test_schema_prefix_domain_and_null():
    s = Schema(3, [Field("a", "int", minimum=1)], (0,))
    assert s.length(s.pad([(2,)])) == 1
    for row in [[(0,), (1,), (0,)], [(1.5,), (0,), (0,)]]:
        with pytest.raises(ValueError):
            s.length(row)
    with pytest.raises(ValueError):
        Schema(2, [Field("x")], (1,), activity=lambda r: True)
    s = Schema(2, [Field("a"), Field("active", "int", choices=(1,))], (0, 0), "active")
    with pytest.raises(ValueError):
        s.length([(2, 1), (5, 0)])


def test_quantizer_boundaries_and_overflow():
    q = LogQuantizer()
    assert q(-0.1) == "<VAL_NEG>"
    assert q(0) == q(-0.0) == "<VAL_ZERO>"
    assert q(0.1) == q(1) == "<VAL_1>"
    assert q(65536) == q(1e100) == "<VAL_65536>"
    for v in np.geomspace(1, 65535, 1000):
        i = max(0, min(199, int(math.log2(max(1, v)) * 200 / 16)))
        assert q(v) == f"<VAL_{int(math.pow(2, i * 16 / 200))}>"
    for bad in [float("nan"), float("inf")]:
        with pytest.raises(ValueError):
            q(bad)
    with pytest.raises(ValueError):
        LogQuantizer(0)


def test_serialization_has_only_active_values_and_positions():
    s, rows, labels = toy()
    serializer = Serializer(s, Quantizer(s))
    records = rows[0][: s.length(rows[0])]
    pieces = serializer.pieces(labels[0], records)
    values = [p for p in pieces if p.role == "value"]
    assert len(values) == len(records) * len(s.fields)
    assert values[0].position == 0 and values[0].field == 0
    assert serializer.text(labels[0], records).endswith("<PPI_END><SAMPLE_END>")


def test_pools_references_and_determinism():
    s, rows, labels = toy()
    p = EmpiricalProposal(s, rows, labels, [0, 1, 2, 3])
    plan = p.plan("demo", 50, k=7, m=3)
    same = p.plan("demo", 50, k=7, m=3)
    assert np.array_equal(plan.candidate_donors, same.candidate_donors)
    assert set(plan.length_donors) <= {0, 1, 2, 3}
    for ri, length in enumerate(plan.lengths):
        assert length == s.length(rows[plan.length_donors[ri]])
        for j in range(length):
            assert all(p.lengths[d] > j for d in plan.candidate_donors[ri, j])
            assert all(p.lengths[d] > j for d in plan.bank_donors[length, :, j])
        assert np.all(plan.candidate_donors[ri, length:] == -1)
    with pytest.raises(ValueError):
        p.plan("missing", 2)


def test_softmax_and_inverse_cdf():
    assert np.array_equal(softmax([1e100, 1e100]), [0.5, 0.5])
    assert np.allclose(softmax([2.0, 4.0]), softmax([102.0, 104.0]))
    assert categorical([0.125] * 8, 0.125) == 1
    assert categorical([0.2, 0.8], np.nextafter(1.0, 0.0)) == 1
    for temp in [0, -1, float("nan")]:
        with pytest.raises(ValueError):
            softmax([0, 1], temp)


def test_raw_values_are_copied_and_rows_pass_validity():
    s, rows, labels = toy()
    p = EmpiricalProposal(s, rows, labels)
    scorer = MarkovScorer(Quantizer(s)).fit(rows, labels, s)
    for rule in ["relative", "uniform", "absolute"]:
        result = Generator(p, scorer, rule).generate("demo", 20)
        assert validate_rows(s, result.rows)
        for ri, row in enumerate(result.rows):
            for j in range(s.length(row)):
                assert row[j] == rows[result.selected_donors[ri, j]][j]
        if rule == "relative":
            assert np.all(result.probabilities[:, 0] == 1 / 8)


def test_markov_exact_backoff_formula():
    s = Schema(2, [Field("x", "int")], (0,))
    q = Quantizer(s)
    scorer = MarkovScorer(q, vocabulary_size=5).fit([[(1,), (2,)]], ["A"], s)
    p0 = (1 + 0.5) / (1 + 0.5 * 5)
    expected = -math.log((1 + 10 * p0) / 11)
    assert scorer.score((2,), [(1,)], "A") == pytest.approx(expected)
    assert scorer.score((2,), [(10,)], "A") == pytest.approx(-math.log(p0))


def test_csv_roundtrip(tmp_path):
    s, rows, labels = toy()
    path = tmp_path / "toy.csv"
    write_csv(path, rows, labels, s, {})
    actual, actual_labels = read_csv(path, s, {})
    assert actual_labels == labels
    assert [[tuple(map(int, r)) for r in row] for row in actual] == rows


def test_cli_quickstart(capsys):
    main(["quickstart", "--rows", "3"])
    assert '"rows": 3' in capsys.readouterr().out


def test_replayed_plan_rejects_ineligible_reference_and_uniform():
    s, rows, labels = toy()
    p = EmpiricalProposal(s, rows, labels)
    scorer = MarkovScorer(Quantizer(s)).fit(rows, labels, s)
    generator = Generator(p, scorer)
    plan = p.plan("demo", 2)
    length = int(plan.lengths[0])
    plan.bank_donors[length, 0, 0] = -1
    with pytest.raises(ValueError, match="reference"):
        generator.generate("demo", 2, plan=plan)
    plan = p.plan("demo", 2)
    plan.selection_uniforms[0, 0] = 1
    with pytest.raises(ValueError, match="uniforms"):
        generator.generate("demo", 2, plan=plan)


def test_empty_rows_and_numeric_text_padding():
    s = Schema(2, [Field("value", "int", minimum=1)], (0,))
    rows = [[("0",), ("0",)], [("2",), ("0",)]]
    p = EmpiricalProposal(s, rows, ["A", "A"])
    result = Generator(p, rule="uniform").generate("A", 30)
    assert {s.length(row) for row in result.rows} == {0, 1}
    assert validate_rows(s, result.rows)


def test_duplicate_csv_headers_rejected(tmp_path):
    schema = Schema(1, [Field("value", "int")], (0,))
    path = tmp_path / "bad.csv"
    path.write_text("value_0,value_0,class\n1,2,A\n")
    with pytest.raises(ValueError, match="duplicate"):
        read_csv(path, schema, {})
