import pytest
import numpy as np

from jinx.vocabulary import Punctuation, Atom, DataType
from jinx.word_evaluation import evaluate_words
from jinx.primitives import PRIMITIVE_MAP


LPAREN = Punctuation("(", name="Left Parenthesis")
RPAREN = Punctuation(")", name="Right Parenthesis")

MINUS = PRIMITIVE_MAP["MINUS"]
PLUS = PRIMITIVE_MAP["PLUS"]


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [Atom(data_type=DataType.Integer, data=1)],
            [Atom(data_type=DataType.Integer, data=1)],
            id="1",
        ),
        pytest.param(
            [LPAREN, Atom(data_type=DataType.Integer, data=1), RPAREN],
            [Atom(data_type=DataType.Integer, data=1)],
            id="(1)",
        ),
        pytest.param(
            [LPAREN, LPAREN, Atom(data_type=DataType.Integer, data=1), RPAREN, RPAREN],
            [Atom(data_type=DataType.Integer, data=1)],
            id="((1))",
        ),
        pytest.param(
            [MINUS],
            [MINUS],
            id="-",
        ),
        pytest.param(
            [LPAREN, MINUS, RPAREN],
            [MINUS],
            id="(-)",
        ),
        pytest.param(
            [MINUS, Atom(data_type=DataType.Integer, data=1)],
            [
                Atom(
                    data_type=DataType.Integer, data=None, implementation=np.int64(-1)
                ),
            ],
            id="-1",
        ),
        pytest.param(
            [MINUS, LPAREN, Atom(data_type=DataType.Integer, data=1), RPAREN],
            [
                Atom(
                    data_type=DataType.Integer, data=None, implementation=np.int64(-1)
                ),
            ],
            id="-(1)",
        ),
        pytest.param(
            [LPAREN, MINUS, RPAREN, LPAREN, Atom(data_type=DataType.Integer, data=1), RPAREN],
            [
                Atom(
                    data_type=DataType.Integer, data=None, implementation=np.int64(-1)
                ),
            ],
            id="(-)(1)",
        ),
        pytest.param(
            [
                Atom(data_type=DataType.Integer, data=1),
                MINUS,
                Atom(data_type=DataType.Integer, data=1),
            ],
            [
                Atom(data_type=DataType.Integer, data=None, implementation=np.int64(0)),
            ],
            id="1-1",
        ),
        pytest.param(
            [
                LPAREN,
                Atom(data_type=DataType.Integer, data=1),
                MINUS,
                Atom(data_type=DataType.Integer, data=1),
                RPAREN,
            ],
            [
                Atom(data_type=DataType.Integer, data=None, implementation=np.int64(0)),
            ],
            id="(1-1)",
        ),
        pytest.param(
            [
                LPAREN,
                Atom(data_type=DataType.Integer, data=8),
                MINUS,
                LPAREN,
                Atom(data_type=DataType.Integer, data=1),
                MINUS,
                Atom(data_type=DataType.Integer, data=5),
                RPAREN,
                RPAREN,
                PLUS,
                Atom(data_type=DataType.Integer, data=3),
            ],
            [
                Atom(
                    data_type=DataType.Integer, data=None, implementation=np.int64(15)
                ),
            ],
            id="(8 - (1 - 5)) + 3",
        ),
        pytest.param(
            [
                LPAREN,
                LPAREN,
                Atom(data_type=DataType.Integer, data=8),
                MINUS,
                Atom(data_type=DataType.Integer, data=1),
                RPAREN,
                MINUS,
                Atom(data_type=DataType.Integer, data=5),
                RPAREN,
                PLUS,
                Atom(data_type=DataType.Integer, data=3),
            ],
            [
                Atom(data_type=DataType.Integer, data=None, implementation=np.int64(5)),
            ],
            id="((8 - 1) - 5) + 3",
        ),
    ],
)
def test_word_evaluation(words, expected):
    result = evaluate_words(words)
    assert result[1:] == expected
