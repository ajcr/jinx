"""Methods for converting to and from NumPy arrays."""

import numpy as np

from jinx.vocabulary import (
    Atom,
    Array,
    DataType,
    Noun,
)


DATATYPE_TO_NP_MAP = {
    DataType.Integer: np.int64,
    DataType.Float: np.float64,
    DataType.Byte: np.str_,
}


def convert_noun_np(noun: Atom | Array) -> np.ndarray:
    dtype = DATATYPE_TO_NP_MAP[noun.data_type]
    return np.array(noun.data, dtype=dtype)


def ensure_noun_implementation(noun: Noun) -> None:
    if noun.implementation is None:
        noun.implementation = convert_noun_np(noun)


def infer_data_type(data):
    try:
        dtype = data.dtype
    except AttributeError:
        dtype = type(data)
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
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


def is_ufunc(func: callable) -> bool:
    return isinstance(func, np.ufunc) or hasattr(func, "ufunc")
