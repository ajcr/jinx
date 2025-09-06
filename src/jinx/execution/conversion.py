"""Methods for converting between J Nouns and NumPy arrays."""

from typing import Any

import numpy as np

from jinx.vocabulary import DataType, Noun

# Define a structured dtype for boxes, which can hold any object.
#
# The alternative of using np.object directly (the 'O' dtype) is problematic for a
# couple of reasons.
#
# Firstly we need to add metadata to the dtype to indicate that it is a box
# because we may also want to use np.object for other purposes (e.g. a rational
# number dtype). Not all NumPy operations preserve the dtype metadata however
# (e.g. np.concatenate), so we would need to patch the metadata back in.
#
# Secondly, np.object presents issues when detecting array sizes and concatenating
# boxed arrays. E.g. with the comma_dyad implemetation that works correct for non-boxed
# arrays, '(<1),(<2 3),(<4)' created a 2D array not a 1D array.
#
# Using a structured dtype allows us to side-step these issues at the small expense
# of making it more difficult to insert and extract data from the box.
box_dtype = np.dtype([("content", "O")])


DATATYPE_TO_NP_MAP = {
    DataType.Integer: np.int64,
    DataType.Float: np.float64,
    DataType.Byte: np.str_,
    DataType.Box: box_dtype,
}


def convert_noun_np(noun: Noun) -> np.ndarray:
    dtype = DATATYPE_TO_NP_MAP[noun.data_type]
    if len(noun.data) == 1:
        # A scalar (ndim == 0) is returned for single element arrays.
        return np.array(noun.data[0], dtype=dtype)
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
    if dtype == box_dtype:
        return DataType.Box

    raise NotImplementedError(f"Cannot handle NumPy dtype: {dtype}")


def ndarray_or_scalar_to_noun(data: np.ndarray) -> Noun:
    data_type = infer_data_type(data)
    return Noun(data_type=data_type, implementation=data)
