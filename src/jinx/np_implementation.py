"""Implementation of Parts of Speech in NumPy.

numba.vectorize is used to create generalised ufuncs which
support accumulation/reduction, broadcasting, etc.
"""

import dataclasses
import operator
from typing import Callable

import numba
import numpy as np

from jinx.vocabulary import (
    Noun,
    Atom,
    Array,
    DataType,
    Verb,
    Adverb,
    Monad,
    Conjunction,
)


INFINITY = float("inf")

DATATYPE_TO_NP_MAP = {
    DataType.Integer: np.int64,
    DataType.Float: np.float64,
    DataType.Byte: np.str_,
}


def convert_noun_np(noun: Atom | Array) -> np.ndarray:
    dtype = DATATYPE_TO_NP_MAP[noun.data_type]
    return np.array(noun.data, dtype=dtype)


def infer_data_type(data):
    try:
        dtype = data.dtype
    except AttributeError:
        dtype = type(data)
    if np.issubdtype(dtype, np.integer):
        return DataType.Integer
    if np.issubdtype(dtype, np.floating):
        return DataType.Float
    if np.issubdtype(dtype, np.character):
        return DataType.Byte
    raise NotImplementedError(f"Cannot handle NumPy dtype: {dtype}")


def format_numeric(n):
    if n < 0:
        return f"_{-n}"
    return f" {n}"


def atom_to_string(atom: Atom) -> str:
    ensure_noun_implementation(atom)
    n = atom.implementation
    if np.isinf(n):
        return "__" if n < 0 else "_"
    if n < 0:
        return f"_{-n}"
    return str(n)


def array_to_string(array: Array) -> str:
    np.set_printoptions(
        infstr="_", formatter={"int_kind": format_numeric, "float_kind": format_numeric}
    )
    ensure_noun_implementation(array)
    return str(array.implementation)


def ndarray_or_scalar_to_noun(data) -> Noun:
    data_type = infer_data_type(data)
    if np.isscalar(data):
        return Atom(data_type=data_type, implementation=data)
    return Array(data_type=data_type, implementation=data)


@numba.vectorize(["float64(int64)", "float64(float64)"])
def percent_monad(y: np.ndarray | int | float) -> np.ndarray:
    """% monad: returns the reciprocal of the array."""
    # N.B. np.reciprocal does not support integer types.
    return np.divide(1, y)


def dollar_monad(arr: np.ndarray | int | float) -> np.ndarray | None:
    """$ monad: returns the shape of the array."""
    if np.isscalar(arr) or arr.size == 1:
        return None
    return np.array(arr.shape)


def dollar_dyad(x: np.ndarray | int | float, y: np.ndarray | int | float) -> np.ndarray:
    """$ dyad: create an array with a particular shape.

    Does not support custom fill values at the moment.
    Does not support INFINITY as an atom of x.
    """
    if np.isscalar(x) and (not np.issubdtype(type(x), np.integer) or x < 0):
        raise ValueError(f"Invalid shape: {x}")

    if np.isscalar(x) or x.size == 1:
        x_shape = (x,)
    else:
        x_shape = tuple(x)

    if np.isscalar(y) or y.size == 1:
        result = np.zeros(x_shape, dtype=x.dtype)
        result[:] = y
        return result

    output_shape = x_shape + y.shape[1:]
    data = y.ravel()
    repeat, fill = divmod(np.prod(output_shape), data.size)
    result = np.concatenate([np.tile(data, repeat), data[:fill]]).reshape(output_shape)
    return result


def idot_monad(arr: np.ndarray | int) -> np.ndarray:
    arr = np.atleast_1d(arr)
    shape = abs(arr)
    n = np.prod(shape)
    axes_to_flip = np.where(arr < 0)[0]
    result = np.arange(n).reshape(shape)
    return np.flip(result, axes_to_flip)


def slash_monad(verb: Verb) -> Callable[[np.ndarray], np.ndarray]:
    if verb.dyad.function is None:
        dyad = PRIMITIVE_MAP[verb.name][1]
    else:
        dyad = verb.dyad.function

    if _is_ufunc(dyad) and verb.dyad.is_commutative:
        return dyad.reduce

    # TODO: generate a ufunc on the fly.
    raise NotImplementedError(
        "Adverb '/' only supports commutative operations with ufuncs for now"
    )


def rank_conjunction(verb: Verb, noun: Atom) -> Verb:
    ensure_noun_implementation(noun)
    rank = noun.implementation
    spelling = f'{verb.spelling}"{rank}'
    return dataclasses.replace(
        verb,
        spelling=spelling,
        name=spelling,
        monad=dataclasses.replace(verb.monad, rank=rank),
    )


PRIMITIVE_MAP = {
    # NAME: (MONDAD, DYAD)
    "EQ": (None, np.equal),
    "MINUS": (operator.neg, np.subtract),
    "PLUS": (np.conj, np.add),
    "STAR": (np.sign, np.multiply),
    "PERCENT": (percent_monad, np.divide),
    "HAT": (np.exp, np.power),
    "DOLLAR": (dollar_monad, dollar_dyad),
    "LTDOT": (np.floor, np.minimum),
    "GTDOT": (np.ceil, np.maximum),
    "IDOT": (idot_monad, None),
    "SLASH": (slash_monad, None),
    "RANK": rank_conjunction,
}


def ensure_noun_implementation(noun: Noun) -> None:
    if noun.implementation is None:
        noun.implementation = convert_noun_np(noun)


def _maybe_pad_with_fill_value(
    arrays: list[np.ndarray], fill_value: int = 0
) -> list[np.ndarray]:
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) == 1:
        return arrays

    if len((len(shape) for shape in shapes)) != 1:
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


def _is_ufunc(func: callable) -> bool:
    return isinstance(func, np.ufunc) or hasattr(func, "ufunc")


def apply_monad(verb: Verb, noun: Noun) -> Noun:
    ensure_noun_implementation(noun)

    # TODO: update primitive map at startup time - ignore
    # if the verb is not a primitive.
    if verb.monad.function is None:
        verb.monad.function = PRIMITIVE_MAP[verb.name][0]

    arr = noun.implementation

    if isinstance(noun, Atom):
        return ndarray_or_scalar_to_noun(verb.monad.function(arr))

    verb_rank = verb.monad.rank
    if verb_rank < 0:
        raise NotImplementedError("Negative verb rank not yet supported")

    noun_rank = noun.implementation.ndim
    r = min(verb_rank, noun_rank)

    # If the verb rank is 0 it applies to each atom of the array.
    # NumPy's unary ufuncs are typically designed to work this way
    # Apply the function directly here as an optimisation.
    if r == 0 and _is_ufunc(verb.monad.function):
        return ndarray_or_scalar_to_noun(verb.monad.function(arr))

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

    cells = [verb.monad.function(cell) for cell in arr_reshaped]
    cells = _maybe_pad_with_fill_value(cells)
    result = np.asarray(cells).reshape(frame_shape + cells[0].shape)
    return ndarray_or_scalar_to_noun(result)


def apply_dyad(verb: Verb, noun_1: Noun, noun_2: Noun) -> Noun:
    # For now, just apply the verb with no regard to its rank.

    ensure_noun_implementation(noun_1)
    ensure_noun_implementation(noun_2)

    # TODO: update primitive map at startup time - ignore
    # if the verb is not a primitive.
    verb.dyad.function = PRIMITIVE_MAP[verb.name][1]
    result = verb.dyad.function(noun_1.implementation, noun_2.implementation)
    return ndarray_or_scalar_to_noun(result)


def apply_adverb_to_verb(verb: Verb, adverb: Adverb) -> Verb:
    if adverb.name in PRIMITIVE_MAP:
        monad_function = PRIMITIVE_MAP[adverb.name][0](verb)
    else:
        raise NotImplementedError(f"Adverb '{adverb.spelling}' not supported")

    spelling = verb.spelling + adverb.spelling

    return Verb(
        spelling=spelling,
        name=spelling,
        monad=Monad(name=spelling, rank=INFINITY, function=monad_function),
    )


# Applying a conjunction usually produces a verb (but not always).
# Assume the simple case here.
def apply_conjunction(
    verb_or_noun_1: Verb | Noun, conjunction: Conjunction, verb_or_noun_2: Verb | Noun
) -> Verb:
    f = PRIMITIVE_MAP[conjunction.name]
    return f(verb_or_noun_1, verb_or_noun_2)
