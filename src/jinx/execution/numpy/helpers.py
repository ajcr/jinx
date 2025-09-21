"""Helper methods for manipulating arrays."""

import itertools
from typing import Callable, Any

import numpy as np

from jinx.execution.numpy.conversion import box_dtype


def get_fill_value(array: np.ndarray) -> int | str:
    """Get the fill value for an array."""
    if np.issubdtype(array.dtype, np.number):
        return 0
    elif np.issubdtype(array.dtype, np.str_):
        return " "
    elif is_box(array):
        return np.array([], dtype=box_dtype).squeeze()
    raise NotImplementedError(f"Fill value for dtype {array.dtype} is not known.")


def maybe_pad_with_fill_value(
    arrays: list[np.ndarray],
    fill_value: Any = None,
) -> list[np.ndarray]:
    """Pad arrays to the same shape with a fill value."""
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) == 1:
        return arrays

    dims = [max(dim) for dim in itertools.zip_longest(*shapes, fillvalue=1)]
    padded_arrays = []

    for arr in arrays:
        if arr.shape == () or np.isscalar(arr):
            arr = np.atleast_1d(arr)

        pad_widths = [(0, dim - shape) for shape, dim in zip(arr.shape, dims)]
        fill_value = fill_value if fill_value is not None else get_fill_value(arr)
        padded_array = np.pad(
            arr, pad_widths, mode="constant", constant_values=fill_value
        )
        padded_arrays.append(padded_array)

    return padded_arrays


def maybe_pad_by_duplicating_atoms(
    arrays: list[np.ndarray],
    fill_value: Any = None,
    ignore_first_dim: bool = True,
) -> list[np.ndarray]:
    """Pad arrays to the same shape, duplicating atoms to fill the required shape.

    Fill values are used to pad arrays of larger shapes.
    """
    is_atom = [np.isscalar(arr) or arr.ndim == 0 for arr in arrays]
    arrays = [np.atleast_1d(arr) for arr in arrays]

    ndim = max(arr.ndim for arr in arrays)
    if ndim == 1:
        ignore_first_dim = False

    reversed_shapes = [arr.shape[::-1] for arr in arrays]

    trailing_dims = [
        max(shape) for shape in itertools.zip_longest(*reversed_shapes, fillvalue=1)
    ]
    trailing_dims.reverse()

    if ignore_first_dim:
        trailing_dims = trailing_dims[1:] or trailing_dims

    padded_arrays = []

    for arr, is_atom_ in zip(arrays, is_atom):
        if is_atom_:
            padded = np.full(trailing_dims, arr, dtype=arr.dtype)

        else:
            arr = increase_ndim(arr, ndim)

            if ignore_first_dim:
                padding = [(0, 0)] + [
                    (0, d - s) for s, d in zip(arr.shape[1:], trailing_dims)
                ]
            else:
                padding = [(0, d - s) for s, d in zip(arr.shape, trailing_dims)]

            fill_value = fill_value if fill_value is not None else get_fill_value(arr)
            padded = np.pad(arr, padding, constant_values=fill_value)

        padded_arrays.append(padded)

    padded_arrays = [increase_ndim(arr, ndim) for arr in padded_arrays]
    return padded_arrays


def maybe_parenthesise_verb_spelling(spelling: str) -> str:
    if spelling.startswith("(") and spelling.endswith(")"):
        return spelling
    return f"({spelling})" if " " in spelling else spelling


def increase_ndim(y: np.ndarray, ndim: int) -> np.ndarray:
    idx = (np.newaxis,) * (ndim - y.ndim) + (slice(None),)
    return y[idx]


def is_box(obj: Any) -> bool:
    return getattr(obj, "dtype", None) == box_dtype


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


def is_ufunc(func: callable) -> bool:
    return isinstance(func, np.ufunc) or hasattr(func, "ufunc")


def mark_ufunc_based(function: Callable[..., Any]) -> bool:
    """Mark a function as a ufunc-based function.

    This is used to identify functions that are typically composed of ufuncs
    and can be applied directly to NumPy arrays by the verb-application methods.

    This greatly speeds up application of some verbs.
    """
    function._is_ufunc_based = True
    return function


def is_ufunc_based(function: Callable[..., Any]) -> bool:
    """Check if a function is a ufunc-based function."""
    return getattr(function, "_is_ufunc_based", False)


def is_same_array(x: np.ndarray, y: np.ndarray) -> bool:
    """Check if two arrays are the same, even if `x is y` is `False` avoiding
    comparison of the array values.

    The arrays are the same if they have the same memory address, shape, strides
    and dtype.
    """
    return x.__array_interface__ == y.__array_interface__
