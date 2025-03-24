import pytest

from word_formation import form_words, Word


@pytest.mark.parametrize(
    ["sentence", "expected_words"],
    [
        pytest.param("", [], id="empty sentence"),
        pytest.param(" ", [], id="whitespace"),
        pytest.param(
            "1",
            [Word(value="1", is_numeric=True, start=0, end=1)],
            id="single digit number",
        ),
        pytest.param(
            "1  ",
            [Word(value="1", is_numeric=True, start=0, end=1)],
            id="single digit number followed by whitespace",
        ),
        pytest.param(
            "1 2",
            [Word(value="1 2", is_numeric=True, start=0, end=3)],
            id="two single digit numbers",
        ),
        pytest.param(
            "1 2 3",
            [Word(value="1 2 3", is_numeric=True, start=0, end=5)],
            id="three single digit numbers",
        ),
        pytest.param(
            "12578",
            [Word(value="12578", is_numeric=True, start=0, end=5)],
            id="five digit number",
        ),
        pytest.param(
            "12x",
            [Word(value="12x", is_numeric=True, start=0, end=3)],
            id="extended precision",
        ),
        pytest.param(
            "125.",
            [Word(value="125.", is_numeric=True, start=0, end=4)],
            id="integer with decimal point",
        ),
        pytest.param("+", [Word(value="+", is_numeric=False, start=0, end=1)], id="+"),
        pytest.param(
            "+.", [Word(value="+.", is_numeric=False, start=0, end=2)], id="+."
        ),
        pytest.param(
            "+:", [Word(value="+:", is_numeric=False, start=0, end=2)], id="+:"
        ),
        pytest.param(
            "+:.", [Word(value="+:.", is_numeric=False, start=0, end=3)], id="+:."
        ),
        pytest.param(
            "- 12 13",
            [
                Word(value="-", is_numeric=False, start=0, end=1),
                Word(value="12 13", is_numeric=True, start=2, end=7),
            ],
            id="- 12 13",
        ),
    ],
)
def test_form_words(sentence, expected_words):
    assert form_words(sentence) == expected_words
