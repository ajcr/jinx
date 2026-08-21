"""Boxes in NumPy, plus helper methods."""

from typing import Any

import numpy as np

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
# boxed arrays. E.g. with the comma_dyad implementation that works correct for non-boxed
# arrays, '(<1),(<2 3),(<4)' created a 2D array not a 1D array.
#
# Using a structured dtype allows us to side-step these issues at the small expense
# of making it more difficult to insert and extract data from the box.
BOX_DTYPE = np.dtype([("content", "O")])


def is_box(obj: Any) -> bool:
    return getattr(obj, "dtype", None) == BOX_DTYPE


def hash_box(array: np.ndarray, level: int = 0) -> int:
    """Compute a hash value for a box array."""
    if not is_box(array):
        raise ValueError("Array must be of box dtype.")

    val = 3331
    for item in array:
        if is_box(item):
            val = (val * 31 + level) % (2**64)
            val ^= hash_box(item, level + 1)
        elif isinstance(item, np.ndarray):
            val ^= hash(item.tobytes())
        else:
            val ^= hash(item)
    return val
