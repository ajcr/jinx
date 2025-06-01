import pytest
import numpy as np

from src.jinx.vocabulary import Array, DataType

from src.jinx.execution.printing import array_to_string


@pytest.mark.parametrize(
    "array, expected",
    [
        pytest.param(
            Array(data_type=DataType.Integer, implementation=np.array([9])),
            "9",
            id="[9]",
        ),
        pytest.param(
            Array(data_type=DataType.Integer, implementation=np.array([1, 2, 3])),
            "1 2 3",
            id="[1, 2, 3]",
        ),
        pytest.param(
            Array(data_type=DataType.Integer, implementation=np.array([1, 2, 3, 4, 5])),
            "1 2 3 4 5",
            id="[1, 2, 3, 4, 5]",
        ),
        pytest.param(
            Array(
                data_type=DataType.Integer, implementation=np.array([1, 2, 3, 4, 5, 6])
            ),
            "1 2 3 4 5 ...",
            id="[1, 2, 3, 4, 5, 6]",
        ),
        pytest.param(
            Array(
                data_type=DataType.Integer, implementation=np.array([[1, 2], [3, 4]])
            ),
            "1 2\n3 4",
            id="[[1, 2], [3, 4]]",
        ),
        pytest.param(
            Array(
                data_type=DataType.Integer, implementation=np.array([[1, 2], [999, 4]])
            ),
            "  1 2\n999 4",
            id="[[1, 2], [999, 4]]",
        ),
        pytest.param(
            Array(
                data_type=DataType.Integer,
                implementation=np.array([[1, 2, 3, 4, 5, 6], [999, 1, 1, 1, 1, 1]]),
            ),
            "  1 2 3 4 5 ...\n999 1 1 1 1 ...",
            id="[[1, 2, 3, 4, 5,6 ], [999, 1, 1, 1, 1, 1]]",
        ),
        pytest.param(
            Array(
                data_type=DataType.Integer, implementation=np.array([[1, 2], [-999, 4]])
            ),
            "   1 2\n_999 4",
            id="[[1, 2], [-999, 4]]",
        ),
        pytest.param(
            Array(
                data_type=DataType.Integer,
                implementation=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            ),
            "1 2\n3 4\n\n5 6\n7 8",
            id="[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
        ),
        pytest.param(
            Array(
                data_type=DataType.Integer,
                implementation=np.array([[[1, 2], [3, 4]], [[5, 6], [999, 8]]]),
            ),
            "  1 2\n  3 4\n\n  5 6\n999 8",
            id="[[[1, 2], [3, 4]], [[5, 6], [999, 8]]]",
        ),
    ],
)
def test_array_to_string(array, expected):
    result = array_to_string(array, max_cols=5)
    assert result == expected
