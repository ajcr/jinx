import dataclasses

import numpy as np
import pytest

from src.jinx.execution.application import maybe_pad_with_fill_value, apply_dyad
from src.jinx.vocabulary import Verb, Array, Dyad, DataType


@pytest.mark.parametrize(
    "arrays, expected",
    [
        pytest.param(
            [np.array([1, 2, 3])],
            [np.array([1, 2, 3])],
            id="[1, 2, 3]",
        ),
        pytest.param(
            [np.array([1, 2, 3]), np.array([7])],
            [np.array([1, 2, 3]), np.array([7, 0, 0])],
            id="[1, 2, 3], [7]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array([[7]])],
            [np.array([[1, 2], [3, 4]]), np.array([[7, 0], [0, 0]])],
            id="[[1, 2], [3, 4]], [[7]]",
        ),
        pytest.param(
            [np.array([[1, 2], [3, 4]]), np.array([[7], [8]])],
            [np.array([[1, 2], [3, 4]]), np.array([[7, 0], [8, 0]])],
            id="[[1, 2], [3, 4]], [[7], [8]]",
        ),
    ],
)
def test_maybe_pad_with_fill_value(arrays, expected):
    result = maybe_pad_with_fill_value(arrays, fill_value=0)
    assert [np.array_equal(arr, expected) for arr in result]


PLUS = Verb(
    name="+",
    spelling="+",
    monad=None,
    dyad=Dyad(name="Add", left_rank=0, right_rank=0, function=np.add),
)


@pytest.mark.parametrize(
    "left_array, right_array, left_rank, right_rank, expected",
    [
        pytest.param(
            np.array([1, 2, 3]),
            np.array([4, 5, 6]),
            0,
            0,
            np.array([5, 7, 9]),
            id="[1, 2, 3] + [4, 5, 6]",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.array([[1, 0], [0, 2]]),
            0,
            0,
            np.array([[2, 2], [3, 6]]),
            id="[[1, 2], [3, 4]] + [[1, 0], [0, 2]]",
        ),
        pytest.param(
            np.array(1),
            np.array([4, 5, 6]),
            0,
            0,
            np.array([5, 6, 7]),
            id="[1] + [4, 5, 6]",
        ),
        pytest.param(
            np.array([4, 5, 6]),
            np.array(1),
            0,
            0,
            np.array([5, 6, 7]),
            id="[4, 5, 6] + [1]",
        ),
        pytest.param(
            np.array(1),
            np.array([[1, 2], [3, 4]]),
            0,
            0,
            np.array([[2, 3], [4, 5]]),
            id="[1] + [[1, 2], [3, 4]]",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.array(1),
            0,
            0,
            np.array([[2, 3], [4, 5]]),
            id="[[1, 2], [3, 4]] + [1]",
        ),
    ],
)
def test_dyadic_application_using_plus(
    left_array, right_array, left_rank, right_rank, expected
):
    """Test the dyadic application of + to two arrays using different ranks."""
    verb = dataclasses.replace(
        PLUS,
        dyad=dataclasses.replace(PLUS.dyad, left_rank=left_rank, right_rank=right_rank),
    )
    left_noun = Array(DataType.Integer, implementation=left_array)
    right_noun = Array(DataType.Integer, implementation=right_array)

    result = apply_dyad(verb, left_noun, right_noun)
    assert np.array_equal(result.implementation, expected)
