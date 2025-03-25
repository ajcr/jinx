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
            id="1",
        ),
        pytest.param(
            "1  ",
            [Word(value="1", is_numeric=True, start=0, end=1)],
            id="1 ",
        ),
        pytest.param(
            "1 2",
            [Word(value="1 2", is_numeric=True, start=0, end=3)],
            id="1 2",
        ),
        pytest.param(
            "1 2 3",
            [Word(value="1 2 3", is_numeric=True, start=0, end=5)],
            id="1 2 3",
        ),
        pytest.param(
            "12578",
            [Word(value="12578", is_numeric=True, start=0, end=5)],
            id="12578",
        ),
        pytest.param(
            "12x",
            [Word(value="12x", is_numeric=True, start=0, end=3)],
            id="12x",
        ),
        pytest.param(
            "125.",
            [Word(value="125.", is_numeric=True, start=0, end=4)],
            id="125.",
        ),
        pytest.param(
            "5 3x _1 _ 5.38 7",
            [Word(value="5 3x _1 _ 5.38 7", is_numeric=True, start=0, end=16)],
            id="5 3x _1 _ 5.38 7",
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
        pytest.param(
            "5 % 7 1 1   +   3",
            [
                Word(value="5", is_numeric=True, start=0, end=1),
                Word(value="%", is_numeric=False, start=2, end=3),
                Word(value="7 1 1", is_numeric=True, start=4, end=9),
                Word(value="+", is_numeric=False, start=12, end=13),
                Word(value="3", is_numeric=True, start=16, end=17),
            ],
            id="5 % 7 1 1   +   3",
        ),
        pytest.param(
            "sum =:+/_6.95*i.3 4",
            [
                Word(value="sum", is_numeric=False, start=0, end=3),
                Word(value="=:", is_numeric=False, start=4, end=6),
                Word(value="+", is_numeric=False, start=6, end=7),
                Word(value="/", is_numeric=False, start=7, end=8),
                Word(value="_6.95", is_numeric=True, start=8, end=13),
                Word(value="*", is_numeric=False, start=13, end=14),
                Word(value="i.", is_numeric=False, start=14, end=16),
                Word(value="3 4", is_numeric=True, start=16, end=19),
            ],
            id="sum =:+/_6.95*i.3 4",
        ),
        pytest.param(
            'v@u"2 x = 23',
            [
                Word(value="v", is_numeric=False, start=0, end=1),
                Word(value="@", is_numeric=False, start=1, end=2),
                Word(value="u", is_numeric=False, start=2, end=3),
                Word(value='"', is_numeric=False, start=3, end=4),
                Word(value="2", is_numeric=True, start=4, end=5),
                Word(value="x", is_numeric=False, start=6, end=7),
                Word(value="=", is_numeric=False, start=8, end=9),
                Word(value="23", is_numeric=True, start=10, end=12),
            ],
            id='v@u"2 x = 23',
        ),
        pytest.param(
            "((2*number) + 7) + const NB. comment",
            [
                Word(value="(", is_numeric=False, start=0, end=1),
                Word(value="(", is_numeric=False, start=1, end=2),
                Word(value="2", is_numeric=True, start=2, end=3),
                Word(value="*", is_numeric=False, start=3, end=4),
                Word(value="number", is_numeric=False, start=4, end=10),
                Word(value=")", is_numeric=False, start=10, end=11),
                Word(value="+", is_numeric=False, start=12, end=13),
                Word(value="7", is_numeric=True, start=14, end=15),
                Word(value=")", is_numeric=False, start=15, end=16),
                Word(value="+", is_numeric=False, start=17, end=18),
                Word(value="const", is_numeric=False, start=19, end=24),
                Word(value="NB. comment", is_numeric=False, start=25, end=36),
            ],
            id="((2*number) + 7) + const NB. comment",
        ),
        pytest.param(
            "  'x' (-~.)/  'string'",
            [
                Word(value="'x'", is_numeric=False, start=2, end=5),
                Word(value="(", is_numeric=False, start=6, end=7),
                Word(value="-", is_numeric=False, start=7, end=8),
                Word(value="~.", is_numeric=False, start=8, end=10),
                Word(value=")", is_numeric=False, start=10, end=11),
                Word(value="/", is_numeric=False, start=11, end=12),
                Word(value="'string'", is_numeric=False, start=14, end=22),
            ],
            id="  'x' (-~.)/  'string'",
        ),
    ],
)
def test_form_words(sentence, expected_words):
    assert form_words(sentence) == expected_words
