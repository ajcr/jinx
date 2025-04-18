import pytest
import numpy as np

from src.jinx.vocabulary import (
    Array,
    DataType,
)
from src.jinx.np_implementation import array_to_string_2, maybe_pad_with_fill_value


@pytest.mark.parametrize(
    "array, expected",
    [
        pytest.param(
            Array(DataType.Integer, implementation=np.array([9])),
            "9",
            id="[9]",
        ),
        pytest.param(
            Array(DataType.Integer, implementation=np.array([1, 2, 3])),
            "1 2 3",
            id="[1, 2, 3]",
        ),
        pytest.param(
            Array(DataType.Integer, implementation=np.array([1, 2, 3, 4, 5])),
            "1 2 3 4 5",
            id="[1, 2, 3, 4, 5]",
        ),
        pytest.param(
            Array(DataType.Integer, implementation=np.array([1, 2, 3, 4, 5, 6])),
            "1 2 3 4 5 ...",
            id="[1, 2, 3, 4, 5, 6]",
        ),
        pytest.param(
            Array(DataType.Integer, implementation=np.array([[1, 2], [3, 4]])),
            "1 2\n3 4",
            id="[[1, 2], [3, 4]]",
        ),
        pytest.param(
            Array(DataType.Integer, implementation=np.array([[1, 2], [999, 4]])),
            "  1 2\n999 4",
            id="[[1, 2], [999, 4]]",
        ),
        pytest.param(
            Array(
                DataType.Integer,
                implementation=np.array([[1, 2, 3, 4, 5, 6], [999, 1, 1, 1, 1, 1]]),
            ),
            "  1 2 3 4 5 ...\n999 1 1 1 1 ...",
            id="[[1, 2, 3, 4, 5,6 ], [999, 1, 1, 1, 1, 1]]",
        ),
        pytest.param(
            Array(DataType.Integer, implementation=np.array([[1, 2], [-999, 4]])),
            "   1 2\n_999 4",
            id="[[1, 2], [-999, 4]]",
        ),
        pytest.param(
            Array(
                DataType.Integer,
                implementation=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            ),
            "1 2\n3 4\n\n5 6\n7 8",
            id="[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
        ),
        pytest.param(
            Array(
                DataType.Integer,
                implementation=np.array([[[1, 2], [3, 4]], [[5, 6], [999, 8]]]),
            ),
            "  1 2\n  3 4\n\n  5 6\n999 8",
            id="[[[1, 2], [3, 4]], [[5, 6], [999, 8]]]",
        ),
    ],
)
def test_array_to_string_2(array, expected):
    result = array_to_string_2(array, max_cols=5)
    assert result == expected


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
