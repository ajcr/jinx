import pytest

from jinx.word_spelling import parse_float, parse_integer


@pytest.mark.parametrize(
    ["value", "expected"],
    [
        pytest.param("0", 0, id="0"),
        pytest.param("00", 0, id="00"),
        pytest.param("0.", 0, id="0."),
        pytest.param("0x", 0, id="0x"),
        pytest.param("0.0", 0, id="0.0"),
        pytest.param("0.00", 0, id="0.00"),
        pytest.param("3.", 3, id="3."),
        pytest.param("378", 378, id="378"),
        pytest.param("667x", 667, id="667x"),
        pytest.param("_9", -9, id="_9"),
        pytest.param("_55", -55, id="_55"),
        pytest.param("_3.", -3, id="_3."),
        pytest.param("_3.0", -3, id="_3.0"),
        pytest.param("_3.00", -3, id="_3.00"),
        pytest.param("_378x", -378, id="_378x"),
        pytest.param("_", None, id="_"),
        pytest.param("__", None, id="__"),
        pytest.param("_x", None, id="_x"),
        pytest.param("_3.1", None, id="_3.1"),
    ],
)
def test_parse_integer(value, expected):
    got = parse_integer(value)
    if expected is not None:
        assert isinstance(got, int)
    assert got == expected


@pytest.mark.parametrize(
    ["value", "expected"],
    [
        pytest.param("0.1", 0.1, id="0.1"),
        pytest.param("00.2", 0.2, id="00.2"),
        pytest.param("_3.14", -3.14, id="_3.14"),
        pytest.param("3.1.2", None, id="3.1.2"),
        pytest.param("3.1x", None, id="3.1x"),
        pytest.param("_", float("inf"), id="_"),
        pytest.param("__", -float("inf"), id="__"),
        pytest.param("_3.1", -3.1, id="_3.1"),
    ],
)
def test_parse_float(value, expected):
    got = parse_float(value)
    if expected is not None:
        assert isinstance(got, float)
    assert got == expected
