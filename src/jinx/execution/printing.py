"""Methods for printing nouns (arrays, atoms)."""

import os

import numpy as np

from jinx.vocabulary import Atom, Array


MAX_COLS = 100


def noun_to_string(array: Atom | Array, max_cols: int = MAX_COLS) -> str:
    arr = np.atleast_1d(array.implementation)
    dtype = arr.dtype

    if arr.shape[-1] > max_cols:
        arr = arr[..., :max_cols]
        append_ellipsis = True
    else:
        append_ellipsis = False

    if np.issubdtype(dtype, np.floating):
        arr = array.implementation.ravel()
        rounded = [format_float(n) for n in arr]
        arr = np.asarray(rounded).reshape(arr.shape)

    # If the array is a boolean array, convert it to int8 for printing.
    if np.issubdtype(dtype, np.bool_):
        arr = arr.view(np.int8)

    arr_str = arr.astype(str)
    arr_str = np.strings.replace(arr_str, "-", "_")
    lengths = np.strings.str_len(arr_str)
    justify = np.max(lengths, axis=tuple(range(arr.ndim - 1)))
    arr_str = np.strings.rjust(arr_str, justify)
    return ndim_n_to_string(arr_str, append_ellipsis=append_ellipsis)


def get_decimal_places(n: float) -> int:
    if n < 1:
        return 6
    if n < 10:
        return 5
    if n < 100:
        return 4
    if n < 1000:
        return 3
    if n < 10000:
        return 2
    if n < 100000:
        return 1
    return 0


def format_float(n: float) -> str:
    if np.isinf(n):
        return "__" if n < 0 else "_"
    sign = "_" if n < 0 else ""
    abs_n = abs(n)
    if abs_n.is_integer():
        return f"{sign}{int(abs_n)}"
    decimal_places = get_decimal_places(abs_n)
    rounded_n = round(n, decimal_places)
    return f"{sign}{rounded_n}"


def ndim_1_to_str(arr: np.ndarray, append_ellipsis: bool) -> str:
    result = " ".join(arr.tolist())
    if append_ellipsis:
        result += " ..."
    return result


def ndim_n_to_string(arr: np.ndarray, append_ellipsis: bool) -> str:
    assert np.issubdtype(arr.dtype, np.character)

    if arr.ndim == 1:
        return ndim_1_to_str(arr, append_ellipsis=append_ellipsis)

    subarrays = []
    for subarr in arr:
        if subarr.ndim == 1:
            subarrays.append(ndim_1_to_str(subarr, append_ellipsis=append_ellipsis))
        else:
            subarrays.append(ndim_n_to_string(subarr, append_ellipsis=append_ellipsis))

    sep = os.linesep * (arr.ndim - 1)
    return sep.join(subarrays)
