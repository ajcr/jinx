"""Helper methods for manipulating arrays."""

import itertools
from typing import Callable, Any

import numpy as np


def maybe_pad_with_fill_value(
    arrays: list[np.ndarray], fill_value: int = 0
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
        padded_array = np.pad(
            arr, pad_widths, mode="constant", constant_values=fill_value
        )
        padded_arrays.append(padded_array)

    return padded_arrays


def maybe_pad_by_duplicating_atoms(
    arrays: list[np.ndarray], fill_value: int = 0
) -> list[np.ndarray]:
    """Pad arrays to the same shape, duplicating atoms to fill the required shape.

    Fill values are used to pad arrays of larger shapes.
    """
    arrays = [np.atleast_1d(arr) for arr in arrays]
    reversed_shape_iters = [reversed(arr.shape) for arr in arrays]
    ndim = max(arr.ndim for arr in arrays)

    trailing_dims = [
        max(shape)
        for shape in itertools.zip_longest(*reversed_shape_iters, fillvalue=1)
    ]
    trailing_dims.reverse()
    trailing_dims = trailing_dims[1:]  # ignore dimension that we will concatenate along

    padded_arrays = []

    for arr in arrays:
        if arr.shape == (1,):
            padded = np.full((1,) + tuple(trailing_dims), arr[0], dtype=arr.dtype)

        else:
            arr = increase_ndim(arr, ndim)
            padded = np.pad(
                arr,
                [(0, 0)] + [(0, d - s) for s, d in zip(arr.shape[1:], trailing_dims)],
                constant_values=fill_value,
            )

        padded_arrays.append(padded)

    return padded_arrays


def maybe_parenthesise_verb_spelling(spelling: str) -> str:
    if spelling.startswith("(") and spelling.endswith(")"):
        return spelling
    return f"({spelling})" if " " in spelling else spelling


def increase_ndim(y: np.ndarray, ndim: int) -> np.ndarray:
    idx = (np.newaxis,) * (ndim - y.ndim) + (slice(None),)
    return y[idx]


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
