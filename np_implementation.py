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


def ndarray_or_scalar_to_noun(data) -> Noun:
    data_type = infer_data_type(data)
    if np.isscalar(data):
        return Atom(data_type=data_type, implementation=data)
    return Array(data_type=data_type, implementation=data)


def eq_np_dyad(x: Noun, y: Noun):
    return np.equal


def minus_np_monad(y: Noun):
    return operator.neg


def add_np_monad(y: Noun):
    return operator.add


def star_np_monad(y: Noun):
    return np.sign


PRIMITIVE_MAP = {
    # NAME: (MONDAD, DYAD)
    "EQ": (None, eq_np_dyad),
    "MINUS": (minus_np_monad, None),
    "ADD": (add_np_monad, None),
    "STAR": (star_np_monad, None),
}
