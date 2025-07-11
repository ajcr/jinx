import pytest
import numpy as np

from jinx.vocabulary import Atom, Array, DataType, Verb
from jinx.word_evaluation import evaluate_words
from jinx.word_spelling import PUNCTUATION_MAP
from jinx.primitives import PRIMITIVE_MAP

## Words
# Punctuation
LPAREN = PUNCTUATION_MAP["("]
RPAREN = PUNCTUATION_MAP[")"]

# Verbs
MINUS = PRIMITIVE_MAP["MINUS"]
PLUS = PRIMITIVE_MAP["PLUS"]
PERCENT = PRIMITIVE_MAP["PERCENT"]
IDOT = PRIMITIVE_MAP["IDOT"]

# Adverbs
SLASH = PRIMITIVE_MAP["SLASH"]

# Conjunctions
RANK = PRIMITIVE_MAP["RANK"]


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [Atom(data_type=DataType.Integer, data=1)],
            [Atom(data_type=DataType.Integer, data=1, implementation=np.array(1))],
            id="1",
        ),
        pytest.param(
            [LPAREN, Atom(data_type=DataType.Integer, data=1), RPAREN],
            [Atom(data_type=DataType.Integer, data=1, implementation=np.array(1))],
            id="(1)",
        ),
        pytest.param(
            [LPAREN, LPAREN, Atom(data_type=DataType.Integer, data=1), RPAREN, RPAREN],
            [Atom(data_type=DataType.Integer, data=1, implementation=np.array(1))],
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
            [
                LPAREN,
                MINUS,
                RPAREN,
                LPAREN,
                Atom(data_type=DataType.Integer, data=1),
                RPAREN,
            ],
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
def test_word_evaluation_basic_arithmetic(words, expected):
    result = evaluate_words(words)
    assert result[1:] == expected


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [PLUS, SLASH],
            "+/",
            id="+/",
        ),
        pytest.param(
            [LPAREN, PLUS, SLASH, RPAREN],
            "+/",
            id="(+/)",
        ),
    ],
)
def test_word_evaluation_adverb_creation(words, expected):
    result = evaluate_words(words)
    assert len(result) == 2
    assert isinstance(result[1], Verb)
    assert result[1].spelling == expected


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [PLUS, SLASH, Atom(data_type=DataType.Integer, data=77)],
            [Atom(data_type=DataType.Integer, data=None, implementation=np.int64(77))],
            id="+/ 77",
        ),
        pytest.param(
            [PLUS, SLASH, Array(data_type=DataType.Integer, data=[1, 3, 5])],
            [Atom(data_type=DataType.Integer, data=None, implementation=np.int64(9))],
            id="+/ 1 3 5",
        ),
        pytest.param(
            [
                LPAREN,
                PLUS,
                SLASH,
                Array(data_type=DataType.Integer, data=[8, 3, 5]),
                RPAREN,
            ],
            [Atom(data_type=DataType.Integer, data=None, implementation=np.int64(16))],
            id="(+/ 8 3 5)",
        ),
        pytest.param(
            [
                LPAREN,
                PLUS,
                SLASH,
                RPAREN,
                Array(data_type=DataType.Integer, data=[8, 3, 5]),
            ],
            [Atom(data_type=DataType.Integer, data=None, implementation=np.int64(16))],
            id="(+/) 8 3 5",
        ),
    ],
)
def test_word_evaluation_adverb_application(words, expected):
    result = evaluate_words(words)
    assert result[1:] == expected


@pytest.mark.parametrize(
    "words, expected_verb_spelling",
    [
        pytest.param(
            [PLUS, RANK, Atom(data_type=DataType.Integer, data=0)],
            '+"0',
            id='+"0',
        ),
        pytest.param(
            [PLUS, RANK, Atom(data_type=DataType.Integer, data=1)],
            '+"1',
            id='+"1',
        ),
        pytest.param(
            [LPAREN, PLUS, RPAREN, RANK, Atom(data_type=DataType.Integer, data=1)],
            '+"1',
            id='(+)"1',
        ),
        pytest.param(
            [PLUS, SLASH, RANK, Atom(data_type=DataType.Integer, data=2)],
            '+/"2',
            id='+/"2',
        ),
        pytest.param(
            [
                PLUS,
                SLASH,
                RANK,
                LPAREN,
                Atom(data_type=DataType.Integer, data=2),
                RPAREN,
            ],
            '+/"2',
            id='+/"(2)',
        ),
        pytest.param(
            [
                PLUS,
                SLASH,
                LPAREN,
                RANK,
                RPAREN,
                Atom(data_type=DataType.Integer, data=2),
            ],
            '+/"2',
            id='+/(")2',
            marks=pytest.mark.xfail,
        ),
    ],
)
def test_word_evaluation_verb_conjunction_noun_application(
    words, expected_verb_spelling
):
    result = evaluate_words(words)
    assert len(result) == 2
    assert isinstance(result[1], Verb)
    assert result[1].spelling == expected_verb_spelling


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [
                PLUS,
                RANK,
                Atom(data_type=DataType.Integer, data=0),
                LPAREN,
                Atom(data_type=DataType.Integer, data=5),
                RPAREN,
            ],
            [Atom(data_type=DataType.Integer, implementation=np.int64(5))],
            id='+"0 (5)',
        ),
        pytest.param(
            [
                LPAREN,
                PLUS,
                RANK,
                Atom(data_type=DataType.Integer, data=0),
                RPAREN,
                Atom(data_type=DataType.Integer, data=5),
            ],
            [Atom(data_type=DataType.Integer, implementation=np.int64(5))],
            id='(+"0) 5',
        ),
        pytest.param(
            [
                LPAREN,
                PLUS,
                RANK,
                Atom(data_type=DataType.Integer, data=0),
                RPAREN,
                LPAREN,
                Atom(data_type=DataType.Integer, data=5),
                RPAREN,
            ],
            [Atom(data_type=DataType.Integer, implementation=np.int64(5))],
            id='(+"0) (5)',
        ),
    ],
)
def test_word_evaluation_verb_conjunction_noun_monad_application(words, expected):
    result = evaluate_words(words)
    assert result[1:] == expected


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [
                PLUS,
                SLASH,
                RANK,
                Atom(data_type=DataType.Integer, data=0),
                LPAREN,
                Atom(data_type=DataType.Integer, data=5),
                RPAREN,
            ],
            [Atom(data_type=DataType.Integer, implementation=np.int64(5))],
            id='+/"0 (5)',
        ),
    ],
)
def test_word_evaluation_verb_adverb_conjunction_noun_monad_application(
    words, expected
):
    result = evaluate_words(words)
    assert result[1:] == expected


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [
                PLUS,
                RANK,
                Atom(data_type=DataType.Integer, data=0),
                PLUS,
                Atom(data_type=DataType.Integer, data=9),
            ],
            [Atom(data_type=DataType.Integer, implementation=np.int64(9))],
            id='+"0 + 9',
        ),
        pytest.param(
            [
                PLUS,
                SLASH,
                RANK,
                Atom(data_type=DataType.Integer, data=0),
                PLUS,
                Atom(data_type=DataType.Integer, data=9),
            ],
            [Atom(data_type=DataType.Integer, implementation=np.int64(9))],
            id='+/"0 + 9',
        ),
    ],
)
def test_word_evaluation_verb_conjunction_noun_verb_monad_application(words, expected):
    result = evaluate_words(words)
    assert result[1:] == expected


@pytest.mark.parametrize(
    "words",
    [
        pytest.param([PLUS, MINUS], id="+-"),
        pytest.param([MINUS, PLUS, SLASH], id="-+/"),
        pytest.param([LPAREN, MINUS, PLUS, RPAREN], id="(-+)"),
        pytest.param([LPAREN, MINUS, PLUS, SLASH, RPAREN], id="(-+/)"),
        pytest.param(
            [LPAREN, LPAREN, MINUS, PLUS, RPAREN, RPAREN],
            id="((-) +)",
        ),
    ],
)
def test_word_evaluation_hook_produces_single_verb(words):
    result = evaluate_words(words)
    assert len(result) == 2
    assert isinstance(result[1], Verb)


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [LPAREN, PLUS, PERCENT, RPAREN, Atom(data_type=DataType.Integer, data=4)],
            Atom(data_type=DataType.Float, implementation=np.float64(4.25)),
            id="(+%)4",
        ),
    ],
)
def test_word_evaluation_hook_correct_result(words, expected):
    result = evaluate_words(words)
    assert len(result) == 2
    assert result[1] == expected


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [
                PRIMITIVE_MAP["TILDEDOT"],
                PRIMITIVE_MAP["BARDOT"],
                LPAREN,
                PRIMITIVE_MAP["PLUSDOT"],
                IDOT,
                RPAREN,
                Atom(data_type=DataType.Integer, data=36),
            ],
            np.array([1, 2, 3, 4, 6, 9, 12, 18, 36]),
            id="~. |. (+. i.) 36",
        ),
        pytest.param(
            [
                PRIMITIVE_MAP["PLUS"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["STARCO"],
                Array(data_type=DataType.Integer, data=[3, 4, 5]),
            ],
            np.array(50),
            id="+/@:*: 3 4 5",
        ),
        # From: https://code.jsoftware.com/wiki/Vocabulary/Modifiers
        pytest.param(
            [
                Atom(data_type=DataType.Integer, data=2),
                PRIMITIVE_MAP["STAR"],
                PRIMITIVE_MAP["PERCENTCO"],
                PRIMITIVE_MAP["AT"],
                PRIMITIVE_MAP["PLUS"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["STARCO"],
                Array(data_type=DataType.Integer, data=[1, 2, 3]),
            ],
            np.array(4.29211),
            id="2 * %: @ + / @: *: 1 2 3",
        ),
        # From: https://code.jsoftware.com/wiki/Vocabulary/Modifiers
        pytest.param(
            [
                Atom(data_type=DataType.Integer, data=2),
                PRIMITIVE_MAP["STAR"],
                PRIMITIVE_MAP["PERCENTCO"],
                PRIMITIVE_MAP["AT"],
                LPAREN,
                PRIMITIVE_MAP["PLUS"],
                PRIMITIVE_MAP["SLASH"],
                RPAREN,
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["STARCO"],
                Array(data_type=DataType.Integer, data=[1, 2, 3]),
            ],
            np.array(7.48331),
            id="2 * %: @ (+ /) @: *: 1 2 3",
        ),
        pytest.param(
            [
                PRIMITIVE_MAP["MINUS"],
                PRIMITIVE_MAP["TILDE"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["IDOT"],
                Atom(data_type=DataType.Integer, data=10),
            ],
            np.array(-27),
            id="-~/i.10",
        ),
        pytest.param(
            [
                PRIMITIVE_MAP["LTCO"],
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["NUMBER"],
                PRIMITIVE_MAP["IDOT"],
                Atom(data_type=DataType.Integer, data=6),
            ],
            np.array(5),
            id="<:@:#i.6",
        ),
        pytest.param(
            [
                LPAREN,
                PRIMITIVE_MAP["LTCO"],
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["NUMBER"],
                RPAREN,
                PRIMITIVE_MAP["IDOT"],
                Atom(data_type=DataType.Integer, data=8),
            ],
            np.array(7),
            id="(<:@:#)i.8",
        ),
        pytest.param(
            [
                Array(data_type=DataType.Integer, data=[1, 2, 3]),
                PRIMITIVE_MAP["PLUS"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["STARCO"],
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["MINUS"],
                Array(data_type=DataType.Integer, data=[2, 2, 2]),
            ],
            np.array(2),
            id="1 2 3 +/@:*:@:- 2 2 2",
        ),
        # See: https://www.reddit.com/r/apljk/comments/1axf4tk/comment/kros5i9/
        pytest.param(
            [
                PRIMITIVE_MAP["PLUS"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["AT"],
                LPAREN,
                PRIMITIVE_MAP["MINUS"],
                PRIMITIVE_MAP["TILDE"],
                PRIMITIVE_MAP["GTDOT"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["BSLASH"],
                PRIMITIVE_MAP["LTDOT"],
                PRIMITIVE_MAP["GTDOT"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["BSLASHDOT"],
                RPAREN,
                Array(
                    data_type=DataType.Integer,
                    data=[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
                ),
            ],
            np.array(6),
            id=r"+/@(-~ >./\ <. >./\.) 0 1 0 2 1 0 1 3 2 1 2 1",
        ),
        # See: https://mmapped.blog/posts/04-square-joy-trapped-rain-water
        pytest.param(
            [
                PRIMITIVE_MAP["PLUS"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["AT"],
                LPAREN,
                LPAREN,
                PRIMITIVE_MAP["GTDOT"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["BSLASH"],
                PRIMITIVE_MAP["LTDOT"],
                PRIMITIVE_MAP["GTDOT"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["BSLASHDOT"],
                RPAREN,
                PRIMITIVE_MAP["MINUS"],
                PRIMITIVE_MAP["SQUARERF"],
                RPAREN,
                Array(
                    data_type=DataType.Integer,
                    data=[0, 1, 0, 2, 1, 0, 1, 3, 2, 1, 2, 1],
                ),
            ],
            np.array(6),
            id=r"+/@((>./\ <. >./\.)-]) 0 1 0 2 1 0 1 3 2 1 2 1",
        ),
        pytest.param(
            [
                LPAREN,
                PRIMITIVE_MAP["COMMA"],
                PRIMITIVE_MAP["RANK"],
                Atom(data_type=DataType.Integer, data=0),
                RPAREN,
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["TILDE"],
                Array(
                    data_type=DataType.Integer,
                    data=[0, 1, 2],
                ),
            ],
            np.array(
                [
                    [[0, 0], [0, 1], [0, 2]],
                    [[1, 0], [1, 1], [1, 2]],
                    [[2, 0], [2, 1], [2, 2]],
                ]
            ),
            id='(,"0)/~ 0 1 2',
        ),
        # Test case where left tine of fork is a noun rather than a verb.
        pytest.param(
            [
                LPAREN,
                PRIMITIVE_MAP["NUMBER"],
                PRIMITIVE_MAP["TILDE"],
                Atom(data_type=DataType.Integer, data=2),
                PRIMITIVE_MAP["BAR"],
                PRIMITIVE_MAP["IDOT"],
                PRIMITIVE_MAP["AT"],
                PRIMITIVE_MAP["NUMBER"],
                RPAREN,
                Array(data_type=DataType.Integer, data=[3, 1, 4, 1, 5, 9, 2]),
            ],
            np.array([1, 1, 9]),
            id="(#~ 2 | i.@#) 3 1 4 1 5 9 2",
        ),
        # Test case where u@v applies u at rank of v (i. has rank 1 by default
        # but must be applied at rank 0, the rank of +).
        pytest.param(
            [
                PRIMITIVE_MAP["IDOT"],
                PRIMITIVE_MAP["AT"],
                PRIMITIVE_MAP["PLUS"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["IDOT"],
                Array(data_type=DataType.Integer, data=[2, 2]),
            ],
            np.array([[0, 1, 0, 0], [0, 1, 2, 3]]),
            id="i.@+/ i. 2 2",
        ),
        pytest.param(
            [
                PRIMITIVE_MAP["BARCO"],
                PRIMITIVE_MAP["BANG"],
                PRIMITIVE_MAP["SLASH"],
                PRIMITIVE_MAP["TILDE"],
                PRIMITIVE_MAP["IDOT"],
                Atom(data_type=DataType.Integer, data=5),
            ],
            np.array(
                [
                    [1, 0, 0, 0, 0],
                    [1, 1, 0, 0, 0],
                    [1, 2, 1, 0, 0],
                    [1, 3, 3, 1, 0],
                    [1, 4, 6, 4, 1],
                ]
            ),
            id="|:!/~i.5",
        ),
        pytest.param(
            [
                PRIMITIVE_MAP["GT"],
                PRIMITIVE_MAP["LT"],
                PRIMITIVE_MAP["RANK"],
                Atom(data_type=DataType.Integer, data=0),
                PRIMITIVE_MAP["SQUARERF"],
                Array(data_type=DataType.Integer, data=[8, 6, 4, 3, 2]),
            ],
            np.array([8, 6, 4, 3, 2]),
            id='> <"0 ] 8 6 4 3 2',
        ),
        pytest.param(
            [
                PRIMITIVE_MAP["SEMI"],
                LPAREN,
                PRIMITIVE_MAP["IDOT"],
                Array(data_type=DataType.Integer, data=[3, 2, 3]),
                RPAREN,
                PRIMITIVE_MAP["SEMI"],
                LPAREN,
                PRIMITIVE_MAP["IDOT"],
                Array(data_type=DataType.Integer, data=[2, 1]),
                RPAREN,
                PRIMITIVE_MAP["SEMI"],
                Atom(data_type=DataType.Integer, data=6),
                PRIMITIVE_MAP["SEMI"],
                Array(data_type=DataType.Integer, data=[9, 2]),
            ],
            np.array(
                [
                    [
                        [0, 1, 2],
                        [3, 4, 5],
                    ],
                    [
                        [6, 7, 8],
                        [9, 10, 11],
                    ],
                    [
                        [12, 13, 14],
                        [15, 16, 17],
                    ],
                    [
                        [0, 0, 0],
                        [1, 0, 0],
                    ],
                    [
                        [6, 6, 6],
                        [6, 6, 6],
                    ],
                    [
                        [9, 2, 0],
                        [0, 0, 0],
                    ],
                ],
            ),
            id=";(i. 3 2 3);(i. 2 1);6;9 2",
        ),
    ],
)
def test_word_evaluation_computes_correct_noun(words, expected):
    result = evaluate_words(words)
    assert len(result) == 2
    assert np.array_equal(np.round(result[1].implementation, 5), expected)


@pytest.mark.parametrize(
    "words, expected",
    [
        pytest.param(
            [PRIMITIVE_MAP["MINUS"], PRIMITIVE_MAP["SLASH"]],
            "-/",
            id="-/",
        ),
        pytest.param(
            [LPAREN, PRIMITIVE_MAP["MINUS"], PRIMITIVE_MAP["SLASH"], RPAREN],
            "-/",
            id="(-/)",
        ),
        pytest.param(
            [
                LPAREN,
                PRIMITIVE_MAP["PERCENT"],
                RPAREN,
                PRIMITIVE_MAP["ATCO"],
                PRIMITIVE_MAP["MINUS"],
                PRIMITIVE_MAP["RANK"],
                Atom(data_type=DataType.Integer, data=0),
            ],
            '%@:-"0',
            id='(%)@:-"0',
        ),
    ],
)
def test_word_evaluation_build_verb(words, expected):
    result = evaluate_words(words)
    assert len(result) == 2
    assert isinstance(result[1], Verb)
    assert result[1].spelling == expected
