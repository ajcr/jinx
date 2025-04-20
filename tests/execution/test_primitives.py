import pytest
import numpy as np

from src.jinx.execution.primitives import comma_dyad


@pytest.mark.parametrize(
    "x, y, expected",
    [
        pytest.param(np.array([1]), np.array([2]), np.array([1, 2]), id="(1),(1)->(2)"),
        pytest.param(
            np.array([1, 2]),
            np.array([3, 4]),
            np.array([1, 2, 3, 4]),
            id="(2),(2)->(4)",
        ),
        pytest.param(
            np.array([1]), np.array([3, 4]), np.array([1, 3, 4]), id="(1),(2)->(3)"
        ),
        pytest.param(
            np.array([9]),
            np.array([[1, 2], [3, 4]]),
            np.array([[9, 9], [1, 2], [3, 4]]),
            id="(1),(2,2)->(3,2)",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.array([9]),
            np.array([[1, 2], [3, 4], [9, 9]]),
            id="(2,2),(1)->(3,2)",
        ),
        pytest.param(
            np.array([7, 8]),
            np.array([[1, 2], [3, 4]]),
            np.array([[7, 8], [1, 2], [3, 4]]),
            id="(2,2),(2)->(3,2)",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.array([7, 8]),
            np.array([[1, 2], [3, 4], [7, 8]]),
            id="(2),(2,2)->(3,2)",
        ),
        pytest.param(
            np.array([[1, 2], [3, 4]]),
            np.arange(9).reshape(3, 3),
            np.array([[1, 2, 0], [3, 4, 0], [0, 1, 2], [3, 4, 5], [6, 7, 8]]),
            id="(2,2),(3,3)->(5,3)",
        ),
        pytest.param(
            np.arange(9).reshape(3, 3),
            np.array([[1, 2], [3, 4]]),
            np.array([[0, 1, 2], [3, 4, 5], [6, 7, 8], [1, 2, 0], [3, 4, 0]]),
            id="(3,3),(2,2)->(5,3)",
        ),
        pytest.param(
            np.arange(8).reshape(2, 4),
            np.arange(9).reshape(3, 3),
            np.array(
                [[0, 1, 2, 3], [4, 5, 6, 7], [0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0]]
            ),
            id="(2,4),(3,3)->(5,4)",
        ),
        pytest.param(
            np.arange(9).reshape(3, 3),
            np.arange(8).reshape(2, 4),
            np.array(
                [[0, 1, 2, 0], [3, 4, 5, 0], [6, 7, 8, 0], [0, 1, 2, 3], [4, 5, 6, 7]]
            ),
            id="(3,3),(2,4)->(5,4)",
        ),
        pytest.param(
            np.arange(4).reshape(2, 2),
            np.arange(9).reshape(1, 3, 3),
            np.array(
                [[[0, 1, 0], [2, 3, 0], [0, 0, 0]], [[0, 1, 2], [3, 4, 5], [6, 7, 8]]]
            ),
            id="(2,2),(1,3,3)->(2,3,3)",
        ),
        pytest.param(
            np.arange(9).reshape(1, 3, 3),
            np.arange(4).reshape(2, 2),
            np.array(
                [[[0, 1, 2], [3, 4, 5], [6, 7, 8]], [[0, 1, 0], [2, 3, 0], [0, 0, 0]]]
            ),
            id="(1,3,3),(2,2)->(2,3,3)",
        ),
    ],
)
def test_comma_dyad(x, y, expected):
    result = comma_dyad(x, y)
    assert np.array_equal(result, expected), f"Expected {expected}, got {result}"
