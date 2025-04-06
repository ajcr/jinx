"""Implementation of Parts of Speech in NumPy."""

import operator

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
    dtype = data.dtype
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
    np.set_printoptions(
        infstr="_", formatter={"int_kind": format_numeric, "float_kind": format_numeric}
    )
    return str(atom.implementation)


def array_to_string(array: Array) -> str:
    np.set_printoptions(
        infstr="_", formatter={"int_kind": format_numeric, "float_kind": format_numeric}
    )
    return str(array.implementation)


def ndarray_or_scalar_to_noun(data) -> Noun:
    data_type = infer_data_type(data)
    if np.isscalar(data):
        return Atom(data_type=data_type, implementation=data)
    return Array(data_type=data_type, implementation=data)


PRIMITIVE_MAP = {
    # NAME: (MONDAD, DYAD)
    "EQ": (None, np.equal),
    "MINUS": (operator.neg, np.subtract),
    "PLUS": (np.conj, np.add),
    "STAR": (np.sign, np.multiply),
    "PERCENT": (np.reciprocal, np.divide),
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

    if r == 0:
        return ndarray_or_scalar_to_noun(verb.monad.function(arr))

    # Verbs apply R-L, for non-commutative operations flip the axis
    # that the verb is applied to.
    if not verb.dyad.is_commutative:
        arr = np.flip(arr, axis=(noun_rank - r))

    try:
        return verb.monad.function(arr, axis=(noun_rank - r))
    except TypeError:
        pass

    result = np.apply_over_axes(verb.monad.function, (noun_rank - r), arr)
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
