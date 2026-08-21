"""Methods for converting between J Nouns and NumPy arrays."""

from typing import Any

import numpy as np
from jinx.execution.numpy.boxes import BOX_DTYPE
from jinx.execution.numpy.printing import array_to_string
from jinx.vocabulary import DataType, Noun

DATATYPE_TO_NP_MAP = {
    DataType.Integer: np.int64,
    DataType.Float: np.float64,
    DataType.Byte: np.str_,
    DataType.Box: BOX_DTYPE,
}


def convert_noun_to_numpy_array(noun: Noun[np.ndarray]) -> np.ndarray:
    dtype = DATATYPE_TO_NP_MAP[noun.data_type]
    if len(noun.data) == 1:
        # A scalar (ndim == 0) is returned for single element arrays.
        return np.array(noun.data[0], dtype=dtype)  # type: ignore[call-overload]
    return np.array(noun.data, dtype=dtype)  # type: ignore[call-overload]


def ensure_noun_implementation(noun: Noun[np.ndarray]) -> None:
    if noun.implementation is None:
        noun.implementation = convert_noun_to_numpy_array(noun)


def infer_data_type(data: np.ndarray) -> DataType:
    dtype = data.dtype
    if np.issubdtype(dtype, np.integer) or np.issubdtype(dtype, np.bool_):
        return DataType.Integer
    if np.issubdtype(dtype, np.floating):
        return DataType.Float
    if np.issubdtype(dtype, np.character):
        return DataType.Byte
    if dtype == BOX_DTYPE:
        return DataType.Box

    raise NotImplementedError(f"Cannot handle NumPy dtype: {dtype}")


def to_verb_str(data: np.ndarray) -> str:
    """String representation for the array if used in a verb, for example left
    or right side of a conjunction."""
    if data.ndim <= 1:
        return array_to_string(data)

    shape = array_to_string(np.array(data.shape))
    return f"({shape}${array_to_string(data.ravel())})"


def ndarray_or_scalar_to_noun(data: np.ndarray) -> Noun[np.ndarray]:
    data_type = infer_data_type(data)
    return Noun[np.ndarray](data_type=data_type, implementation=np.asarray(data))


def convert_python_object_to_noun(obj: Any) -> Noun[np.ndarray] | None:
    if isinstance(obj, np.ndarray):
        return ndarray_or_scalar_to_noun(obj)
    if isinstance(obj, (int, float, str)):
        # Wrap Python scalars in a 0-dim NumPy array.
        np_type = DATATYPE_TO_NP_MAP[
            DataType.Integer
            if isinstance(obj, int)
            else DataType.Float
            if isinstance(obj, float)
            else DataType.Byte
        ]
        array = np.asarray(obj, dtype=np_type)  # type: ignore[call-overload]
        return ndarray_or_scalar_to_noun(array)

    return None
