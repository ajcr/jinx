"""Implementation of Parts of Speech in NumPy."""

import operator

import numpy as np

from vocabulary import Noun, Atom, Array, DataType


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
    return str(n)


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


def eq_np_dyad(x: Noun, y: Noun):
    return np.equal


def minus_np_monad(y: Noun):
    return operator.neg


def minus_np_dyad(x: Noun, y: Noun):
    return np.subtract


def add_np_monad(y: Noun):
    return operator.add


def add_np_dyad(x: Noun, y: Noun):
    return np.add


def star_np_monad(y: Noun):
    return np.sign


def star_np_dyad(x: Noun, y: Noun):
    return np.multiply


def percent_np_monad(y: Noun):
    return np.reciprocal


def percent_np_dyad(x: Noun, y: Noun):
    return np.divide


PRIMITIVE_MAP = {
    # NAME: (MONDAD, DYAD)
    "EQ": (None, eq_np_dyad),
    "MINUS": (minus_np_monad, minus_np_dyad),
    "PLUS": (add_np_monad, add_np_dyad),
    "STAR": (star_np_monad, star_np_dyad),
    "PERCENT": (percent_np_monad, percent_np_dyad),
}
