"""Methods for applying verb implementations to nouns and verbs.

Main references:
* https://code.jsoftware.com/wiki/Vocabulary/Agreement
* https://www.jsoftware.com/help/jforc/loopless_code_i_verbs_have_r.htm

"""

import functools

import numpy as np

from jinx.vocabulary import Noun, Verb, Conjunction, Adverb, Monad, Dyad
from jinx.errors import LengthError
from jinx.execution.conversion import ndarray_or_scalar_to_noun, is_ufunc
from jinx.execution.primitives import PRIMITIVE_MAP


def maybe_pad_with_fill_value(
    arrays: list[np.ndarray], fill_value: int = 0
) -> list[np.ndarray]:
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) == 1:
        return arrays

    if len({len(shape) for shape in shapes}) != 1:
        raise NotImplementedError("Cannot pad arrays of different ranks")

    dims = [max(dim) for dim in zip(*shapes)]
    padded_arrays = []

    for arr in arrays:
        pad_widths = [(0, dim - shape) for shape, dim in zip(arr.shape, dims)]
        padded_array = np.pad(
            arr, pad_widths, mode="constant", constant_values=fill_value
        )
        padded_arrays.append(padded_array)

    return padded_arrays


def apply_monad(verb: Verb, noun: Noun) -> np.ndarray:
    arr = noun.implementation

    result = _apply_monad(verb, arr)
    return ndarray_or_scalar_to_noun(result)


def _apply_monad(verb: Verb, arr: np.ndarray) -> np.ndarray:
    if isinstance(verb.monad.function, Verb):
        function = functools.partial(_apply_monad, verb.monad.function)
    else:
        function = verb.monad.function

    verb_rank = verb.monad.rank
    if verb_rank < 0:
        raise NotImplementedError("Negative verb rank not yet supported")

    noun_rank = arr.ndim
    r = min(verb_rank, noun_rank)

    # If the verb rank is 0 it applies to each atom of the array.
    # NumPy's unary ufuncs are typically designed to work this way
    # Apply the function directly here as an optimisation.
    if r == 0 and is_ufunc(function):
        return function(arr)

    # Look at the shape of the array and the rank of the verb to
    # determine the frame and cell shape.
    #
    # The trailing r axes define the cell shape and the preceding
    # axes define the frame shape. E.g. for r=2:
    #
    #   arr.shape = (n0, n1, n2, n3, n4)
    #                ----------  ------
    #                ^ frame     ^ cell
    #
    # If r=0, the frame shape is the same as the shape and the monad
    # applies to each atom of the array.
    if r == 0:
        frame_shape = arr.shape
        arr_reshaped = arr.ravel()
    else:
        cell_shape = arr.shape[-r:]
        frame_shape = arr.shape[:-r]
        arr_reshaped = arr.reshape(-1, *cell_shape)

    cells = [function(cell) for cell in arr_reshaped]
    if len(cells) == 1 and np.isscalar(cells[0]):
        result = cells[0]
    else:
        cells = maybe_pad_with_fill_value(cells)
        result = np.asarray(cells).reshape(frame_shape + cells[0].shape)
    return result


def apply_dyad(verb: Verb, noun_1: Noun, noun_2: Noun) -> Noun:
    left_arr = noun_1.implementation
    right_arr = noun_2.implementation

    result = _apply_dyad(verb, left_arr, right_arr)
    return ndarray_or_scalar_to_noun(result)


def _apply_dyad(verb: Verb, left_arr: np.ndarray, right_arr: np.ndarray) -> np.ndarray:
    # If the dyad is another Verb object, apply_dyad will be called
    # recursively. This allows for implicit nested loops given by
    # verb definitions such as `(+"1 2)"0 3`.
    #
    # Otherwise, the dyad function is a callable implementing a J
    # primitive and is applied directly.
    if isinstance(verb.dyad.function, Verb):
        function = functools.partial(_apply_dyad, verb.dyad.function)
    else:
        function = verb.dyad.function

    left_rank = min(left_arr.ndim, verb.dyad.left_rank)
    right_rank = min(right_arr.ndim, verb.dyad.right_rank)

    if left_rank < 0:
        raise NotImplementedError("Negative left rank not yet supported")

    if right_rank < 0:
        raise NotImplementedError("Negative right rank not yet supported")

    # Cell and frame shapes are determined using the same approach as for
    # the monadic application above.
    if left_rank == 0:
        left_cell_shape = (1,)
        left_frame_shape = left_arr.shape
        left_arr_reshaped = left_arr.ravel()
    else:
        left_cell_shape = left_arr.shape[-left_rank:]
        left_frame_shape = left_arr.shape[:-left_rank]
        left_arr_reshaped = left_arr.reshape(-1, *left_cell_shape)

    if right_rank == 0:
        right_cell_shape = (1,)
        right_frame_shape = right_arr.shape
        right_arr_reshaped = right_arr.ravel()
    else:
        right_cell_shape = right_arr.shape[-right_rank:]
        right_frame_shape = right_arr.shape[:-right_rank]
        right_arr_reshaped = right_arr.reshape(-1, *right_cell_shape)

    # If the frame shapes are the same, we can apply the dyad without further manipulation.
    if left_frame_shape == right_frame_shape:
        cells = [
            function(left_cell, right_cell)
            for left_cell, right_cell in zip(
                left_arr_reshaped, right_arr_reshaped, strict=True
            )
        ]
        cells = maybe_pad_with_fill_value(cells)
        result = np.asarray(cells).reshape(left_frame_shape + cells[0].shape)
        return result

    # Otherwise we need to find the common frame shape. One of the frame shapes must
    # be a prefix of the other, otherwise it's not possible to apply the dyad.
    common_frame_shape = find_common_frame_shape(left_frame_shape, right_frame_shape)
    if common_frame_shape is None:
        raise LengthError(
            f"Cannot apply dyad {verb.spelling} to arrays of shape {left_frame_shape} and {right_frame_shape}"
        )

    rcf = len(common_frame_shape)
    left_rcf_cell_shape = left_arr.shape[rcf:] or (1,)
    right_rcf_cell_shape = right_arr.shape[rcf:] or (1,)

    left_arr_reshaped = left_arr.reshape(-1, *left_rcf_cell_shape)
    right_arr_reshaped = right_arr.reshape(-1, *right_rcf_cell_shape)

    cells = []
    for left_cell, right_cell in zip(
        left_arr_reshaped, right_arr_reshaped, strict=True
    ):
        subcells = []
        if common_frame_shape == left_frame_shape:
            # right_cell is longer and contains multiple operand cells
            r = (
                right_cell.reshape(-1, *right_cell_shape)
                if right_rank > 0
                else right_cell.ravel()
            )
            for right_subcell in r:
                subcells.append(function(left_cell, right_subcell))
        else:
            # left_cell is longer and contains multiple operand cells
            l = (
                left_cell.reshape(-1, *left_cell_shape)
                if left_rank > 0
                else left_cell.ravel()
            )
            for left_subcell in l:
                subcells.append(function(left_subcell, right_cell))

        subcells = maybe_pad_with_fill_value(subcells)
        cells.append(np.asarray(subcells))

    cells = maybe_pad_with_fill_value(cells)
    cells = np.asarray(cells)

    final_frame_shape = max(left_frame_shape, right_frame_shape, key=len)

    try:
        return cells.reshape(final_frame_shape)
    except ValueError:
        return cells


def find_common_frame_shape(
    left_frame_shape: tuple[int, ...], right_frame_shape: tuple[int, ...]
) -> tuple[int, ...] | None:
    if len(left_frame_shape) <= len(right_frame_shape):
        shorter = left_frame_shape
        longer = right_frame_shape
    else:
        shorter = right_frame_shape
        longer = left_frame_shape

    if all(a == b for a, b in zip(shorter, longer)):
        return shorter

    return None


# Applying a conjunction usually produces a verb (but not always).
# Assume that it's just a verb for now.
def apply_conjunction(
    verb_or_noun_1: Verb | Noun, conjunction: Conjunction, verb_or_noun_2: Verb | Noun
) -> Verb:
    f = PRIMITIVE_MAP[conjunction.name]
    return f(verb_or_noun_1, verb_or_noun_2)


def apply_adverb_to_verb(verb: Verb, adverb: Adverb) -> Verb:
    if adverb.name in PRIMITIVE_MAP:
        function = PRIMITIVE_MAP[adverb.name]
    else:
        raise NotImplementedError(f"Adverb '{adverb.spelling}' not supported")
    return function(verb)


def ensure_verb_implementation(verb: Verb) -> None:
    if verb.monad and verb.monad.function is None and verb.name in PRIMITIVE_MAP:
        verb.monad.function = PRIMITIVE_MAP[verb.name][0]
    if verb.dyad and verb.dyad.function is None and verb.name in PRIMITIVE_MAP:
        verb.dyad.function = PRIMITIVE_MAP[verb.name][1]


INFINITY = float("inf")


def build_hook(f: Verb, g: Verb) -> Verb:
    def _monad(y: np.ndarray) -> np.ndarray:
        a = g.monad.function(y)
        return f.dyad.function(y, a)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = g.monad.function(y)
        return f.dyad.function(x, a)

    f_spelling = f"({f.spelling})" if " " in f.spelling else f.spelling
    g_spelling = f"({g.spelling})" if " " in g.spelling else g.spelling

    spelling = f"{f_spelling} {g_spelling}"

    return Verb(
        spelling=spelling,
        name=spelling,
        monad=Monad(
            name=spelling,
            rank=INFINITY,
            function=_monad,
        ),
        dyad=Dyad(
            name=spelling,
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=_dyad,
        ),
    )


def build_fork(f: Verb, g: Verb, h: Verb) -> Verb:
    def _monad(y: np.ndarray) -> np.ndarray:
        a = f.monad.function(y)
        b = h.monad.function(y)
        return g.dyad.function(a, b)

    def _dyad(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        a = f.dyad.function(x, y)
        b = h.dyad.function(x, y)
        return g.dyad.function(a, b)

    f_spelling = f"({f.spelling})" if " " in f.spelling else f.spelling
    g_spelling = f"({g.spelling})" if " " in g.spelling else g.spelling
    h_spelling = f"({h.spelling})" if " " in h.spelling else h.spelling

    spelling = f"{f_spelling} {g_spelling} {h_spelling}"

    return Verb(
        spelling=spelling,
        name=spelling,
        monad=Monad(
            name=spelling,
            rank=INFINITY,
            function=_monad,
        ),
        dyad=Dyad(
            name=spelling,
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=_dyad,
        ),
    )
