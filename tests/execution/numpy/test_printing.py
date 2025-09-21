import pytest
import numpy as np

from jinx.vocabulary import Noun, DataType
from jinx.execution.numpy.printing import noun_to_string
from jinx.execution.numpy.verbs import lt_monad, semi_dyad, dollar_dyad


@pytest.mark.parametrize(
    "array, expected",
    [
        pytest.param(
            Noun(data_type=DataType.Integer, implementation=np.array([9])),
            "9",
            id="[9]",
        ),
        pytest.param(
            Noun(data_type=DataType.Integer, implementation=np.array([1, 2, 3])),
            "1 2 3",
            id="[1, 2, 3]",
        ),
        pytest.param(
            Noun(data_type=DataType.Integer, implementation=np.array([1, 2, 3, 4, 5])),
            "1 2 3 4 5",
            id="[1, 2, 3, 4, 5]",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Integer, implementation=np.array([1, 2, 3, 4, 5, 6])
            ),
            "1 2 3 4 5 ...",
            id="[1, 2, 3, 4, 5, 6]",
        ),
        pytest.param(
            Noun(data_type=DataType.Integer, implementation=np.array([[1, 2], [3, 4]])),
            "1 2\n3 4",
            id="[[1, 2], [3, 4]]",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Integer, implementation=np.array([[1, 2], [999, 4]])
            ),
            "  1 2\n999 4",
            id="[[1, 2], [999, 4]]",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Integer,
                implementation=np.array([[1, 2, 3, 4, 5, 6], [999, 1, 1, 1, 1, 1]]),
            ),
            "  1 2 3 4 5 ...\n999 1 1 1 1 ...",
            id="[[1, 2, 3, 4, 5,6 ], [999, 1, 1, 1, 1, 1]]",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Integer, implementation=np.array([[1, 2], [-999, 4]])
            ),
            "   1 2\n_999 4",
            id="[[1, 2], [-999, 4]]",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Integer,
                implementation=np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]]),
            ),
            "1 2\n3 4\n\n5 6\n7 8",
            id="[[[1, 2], [3, 4]], [[5, 6], [7, 8]]]",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Integer,
                implementation=np.array([[[1, 2], [3, 4]], [[5, 6], [999, 8]]]),
            ),
            "  1 2\n  3 4\n\n  5 6\n999 8",
            id="[[[1, 2], [3, 4]], [[5, 6], [999, 8]]]",
        ),
    ],
)
def test_noun_to_string(array, expected):
    result = noun_to_string(array, max_cols=5)
    assert result == expected


@pytest.mark.parametrize(
    "array, expected",
    [
        pytest.param(
            Noun(data_type=DataType.Box, implementation=lt_monad(np.array(9))),
            "┌─┐\n│9│\n└─┘",
            id="<9",
        ),
        pytest.param(
            Noun(data_type=DataType.Box, implementation=lt_monad(np.array([9]))),
            "┌─┐\n│9│\n└─┘",
            id="<9",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Box, implementation=lt_monad(lt_monad(np.array(3)))
            ),
            "┌───┐\n│┌─┐│\n││3││\n│└─┘│\n└───┘",
            id="<<3",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Box,
                implementation=semi_dyad(lt_monad(np.array(3)), np.array(5)),
            ),
            "┌───┬─┐\n│┌─┐│5│\n││3││ │\n│└─┘│ │\n└───┴─┘",
            id="(<3);5",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Box,
                implementation=dollar_dyad(
                    np.array([2, 2]),
                    semi_dyad(
                        np.array(1000),
                        semi_dyad(np.array(1), semi_dyad(np.array(10), np.array(100))),
                    ),
                ),
            ),
            "┌────┬───┐\n│1000│1  │\n├────┼───┤\n│10  │100│\n└────┴───┘",
            id="2 2 $ 1000;1;10;100",
        ),
        pytest.param(
            Noun(
                data_type=DataType.Box,
                implementation=semi_dyad(
                    np.array(list("alpha")),
                    semi_dyad(np.array(list("bravo")), np.array(list("charlie"))),
                ),
            ),
            "┌─────┬─────┬───────┐\n│alpha│bravo│charlie│\n└─────┴─────┴───────┘",
            id="'alpha' ; 'bravo' ; 'charlie'",
        ),
    ],
)
def test_boxed_noun_to_string(array, expected):
    result = noun_to_string(array, max_cols=5)
    assert result == expected
