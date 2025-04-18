import pytest
import numpy as np

from src.jinx.vocabulary import (
    Array,
    DataType,
)
from src.jinx.np_implementation import array_to_string_2


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
