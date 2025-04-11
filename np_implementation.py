"""Implementation of Parts of Speech in NumPy.

numba.vectorize is used to create generalised ufuncs which
support accumulation/reduction, broadcasting, etc.
"""

import operator

import numba
import numpy as np

from vocabulary import Noun, Atom, Array, DataType, Verb


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
    if np.isscalar(arr):
        return None
    return np.array(arr.shape, dtype=np.int64)


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


PRIMITIVE_MAP = {
    # NAME: (MONDAD, DYAD)
    "EQ": (None, np.equal),
    "MINUS": (operator.neg, np.subtract),
    "PLUS": (np.conj, np.add),
    "STAR": (np.sign, np.multiply),
    "PERCENT": (percent_monad, np.divide),
    "DOLLAR": (dollar_monad, dollar_dyad),
}


def ensure_noun_implementation(noun: Noun) -> None:
    if noun.implementation is None:
        noun.implementation = convert_noun_np(noun)


def apply_monad(verb: Verb, noun: Noun) -> Noun:
    ensure_noun_implementation(noun)

    # TODO: update primitive map at startup time - ignore
    # if the verb is not a primitive.
    verb.monad.function = PRIMITIVE_MAP[verb.name][0]

    if isinstance(noun, Atom):
        return ndarray_or_scalar_to_noun(verb.monad.function(noun.implementation))

    verb_rank = verb.monad.rank
    noun_rank = noun.implementation.ndim

    r = min(verb_rank, noun_rank)
    arr = noun.implementation

    # If the verb rank is 0 it applies to each atom of the array.
    # NumPy's unary ufuncs are typically designed to work this way.
    if r == 0:
        return ndarray_or_scalar_to_noun(verb.monad.function(arr))

    # Applying a verb of rank R to an array is roughly the same as
    # applying the function along an axis of the array.
    axis = noun_rank - r

    try:
        return verb.monad.function(arr, axis=axis)
    except TypeError:
        pass

    result = np.apply_over_axes(verb.monad.function, axis, arr)
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
