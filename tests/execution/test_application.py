import numpy as np
import pytest

from jinx.errors import LengthError
from src.jinx.execution.application import apply_dyad
from src.jinx.vocabulary import Noun, DataType
from src.jinx.execution.primitives import (
    PRIMITIVE_MAP as PRIMITIVE_MAP_NP,
)
from src.jinx.execution.conjunctions import rank_conjunction
from src.jinx.primitives import PRIMITIVE_MAP


PLUS = PRIMITIVE_MAP["PLUS"]
PLUS.monad.function = PRIMITIVE_MAP_NP["PLUS"][0]
PLUS.dyad.function = PRIMITIVE_MAP_NP["PLUS"][1]
PLUS_0_1 = rank_conjunction(
    PLUS, Noun(data_type=DataType.Integer, implementation=np.array([0, 1]))
)
PLUS_1_0 = rank_conjunction(
    PLUS, Noun(data_type=DataType.Integer, implementation=np.array([1, 0]))
)
PLUS_0_2 = rank_conjunction(
    PLUS, Noun(data_type=DataType.Integer, implementation=np.array([0, 2]))
)
PLUS_1_2 = rank_conjunction(
    PLUS, Noun(data_type=DataType.Integer, implementation=np.array([1, 2]))
)


@pytest.mark.parametrize(
    "verb, left_array, right_array, expected",
    [
        pytest.param(
            PLUS,
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            np.array([5, 7, 9]),
            id="1 2 3 + 4 5 6",
        ),
        pytest.param(
            PLUS,
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 0], [0, 2]]),
            np.array([[2, 2], [3, 6]]),
            id="(>: i. 2 2) + (2 2 $ 1 0 0 2)",
        ),
        pytest.param(
            PLUS,
            np.array(1),
            np.array([4, 5, 6]),
            np.array([5, 6, 7]),
            id="1 + 4 5 6",
        ),
        pytest.param(
            PLUS,
            np.array([4, 5, 6]),
            np.array(1),
            np.array([5, 6, 7]),
            id="4 5 6 + 1",
        ),
        pytest.param(
            PLUS,
            np.array([4, 5, 6]),
            np.array([5, 6, 7]),
            np.array([9, 11, 13]),
            id="4 5 6 + 5 6 7",
        ),
        pytest.param(
            PLUS,
            np.array(1),
            np.array([[1, 2], [3, 4]]),
            np.array([[2, 3], [4, 5]]),
            id="1 + (>: i. 2 2)",
        ),
        pytest.param(
            PLUS,
            np.array([[1, 2], [3, 4]]),
            np.array(1),
            np.array([[2, 3], [4, 5]]),
            id="(>: i. 2 2) + 1",
        ),
        pytest.param(
            PLUS_0_1,
            np.array([0, 100]),
            np.arange(6).reshape(2, 3),
            np.array([[0, 1, 2], [103, 104, 105]]),
            id='(0 100) +"0 1 (i. 2 3)',
        ),
        pytest.param(
            PLUS_0_1,
            np.array([0, 100]),
            np.arange(6).reshape(2, 3, 1),
            np.array([[[0], [1], [2]], [[103], [104], [105]]]),
            id='(0 100) +"0 1 (i. 2 3 1)',
        ),
        pytest.param(
            PLUS_1_0,
            np.array([0, 100]),
            np.arange(8).reshape(2, 2, 2),
            np.array(
                [
                    [[[0, 100], [1, 101]], [[2, 102], [3, 103]]],
                    [[[4, 104], [5, 105]], [[6, 106], [7, 107]]],
                ]
            ),
            id='(0 100) +"1 0 (i. 2 2 2)',
        ),
        pytest.param(
            PLUS_0_1,
            np.array([0, 500]),
            np.arange(8).reshape(2, 2, 2),
            np.array([[[0, 1], [2, 3]], [[504, 505], [506, 507]]]),
            id='(0 500) +"0 1 (i. 2 2 2)',
        ),
        pytest.param(
            PLUS_0_2,
            np.arange(6).reshape(3, 2),
            np.arange(6).reshape(3, 1, 2),
            np.array(
                [[[[0, 1]], [[1, 2]]], [[[4, 5]], [[5, 6]]], [[[8, 9]], [[9, 10]]]]
            ),
            id='(0 500) +"0 1 (i. 2 2 2)',
        ),
    ],
)
def test_dyadic_application_using_plus(verb, left_array, right_array, expected):
    """Test the dyadic application of + to two arrays using different ranks."""
    left_noun = Noun(data_type=DataType.Integer, implementation=left_array)
    right_noun = Noun(data_type=DataType.Integer, implementation=right_array)

    result = apply_dyad(verb, left_noun, right_noun)
    assert np.array_equal(result.implementation, expected)


@pytest.mark.parametrize(
    "verb, left_array, right_array",
    [
        pytest.param(
            PLUS,
            np.array([0, 500]),
            np.array([1, 2, 3]),
            id="0 500 + 1 2 3",
        ),
    ],
)
def test_dyadic_application_using_plus_raises_length_error(
    verb, left_array, right_array
):
    left_noun = Noun(data_type=DataType.Integer, implementation=left_array)
    right_noun = Noun(data_type=DataType.Integer, implementation=right_array)
    with pytest.raises(LengthError):
        apply_dyad(verb, left_noun, right_noun)
