"""Methods for printing nouns (arrays, atoms)."""

import os

import numpy as np

from jinx.vocabulary import Atom, Array


def atom_to_string(atom: Atom) -> str:
    n = atom.implementation
    return format_float(n) if isinstance(n, float) else str(n)


def rank_1_array_integer_to_string(
    arr: np.ndarray, justify: list[int], append_ellisis: bool = False
) -> str:
    assert arr.ndim == 1
    assert len(justify) == len(arr)

    scalars = []

    for scalar, pad in zip(arr, justify):
        sign = "_" if scalar < 0 else ""
        str_ = f"{sign}{abs(scalar)}"
        str_ = str_.rjust(pad)
        scalars.append(str_)

    if append_ellisis:
        scalars.append("...")

    return " ".join(scalars)


def rank_n_array_string(
    arr: np.ndarray, justify: list[int], append_ellisis: bool = False
) -> str:
    if arr.ndim == 1:
        return rank_1_array_integer_to_string(
            arr, justify, append_ellisis=append_ellisis
        )

    subarray_strs = []

    for subarr in arr:
        if subarr.ndim == 1:
            str_ = rank_1_array_integer_to_string(
                subarr, justify, append_ellisis=append_ellisis
            )
        else:
            str_ = rank_n_array_string(subarr, justify, append_ellisis=append_ellisis)

        subarray_strs.append(str_)

    sep = os.linesep * (arr.ndim - 1)
    return sep.join(subarray_strs)


MAX_COLS = 100


def array_to_string(array: Array, max_cols: int = MAX_COLS) -> str:
    arr = np.atleast_1d(array.implementation)
    ndim = arr.ndim
    dtype = arr.dtype

    if arr.shape[-1] > max_cols:
        arr = arr[..., :max_cols]
        append_ellipsis = True
    else:
        append_ellipsis = False

    # For now, just use NumPy for printing float arrays
    if np.issubdtype(dtype, np.floating):
        return float_array_to_string(array)

    # If the array is a boolean array, convert it to int8 for printing.
    if np.issubdtype(dtype, np.bool_):
        arr = arr.view(np.int8)

    arr_abs = np.abs(arr)
    ndigits = np.zeros_like(arr_abs, dtype=np.float64)
    np.log10(arr_abs, out=ndigits, where=(arr_abs != 0))
    ndigits = ndigits.astype(int) + 1

    reduce_over = tuple(range(ndim - 1))
    columnwise_max_digits = np.max(ndigits, axis=reduce_over)

    # Check if a value with max_digits for a column is also negative.
    # If so, add 1 to compensate for the sign.
    is_negative_with_max_digits = np.logical_and(
        arr < 0, ndigits == columnwise_max_digits
    )
    columnwise_negative_with_max_digits = np.any(
        is_negative_with_max_digits, axis=reduce_over
    )
    columnwise_max_digits[columnwise_negative_with_max_digits] += 1

    justify = columnwise_max_digits.tolist()

    return rank_n_array_string(arr, justify, append_ellisis=append_ellipsis)


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


def float_array_to_string(array: Array) -> str:
    arr = array.implementation.ravel()
    rounded = [format_float(n) for n in arr]
    arr = np.asarray(rounded).reshape(array.implementation.shape)

    lengths = np.strings.str_len(arr)
    justify = np.max(lengths, axis=tuple(range(arr.ndim - 1)))

    arr = np.strings.rjust(arr, justify)
    return ndim_n_to_string(arr)


def ndim_1_to_str(arr: np.ndarray) -> str:
    return " ".join(arr.tolist())


def ndim_n_to_string(arr: np.ndarray) -> str:
    assert np.issubdtype(arr.dtype, np.character)

    if arr.ndim == 1:
        return ndim_1_to_str(arr)

    subarrays = []
    for subarr in arr:
        if subarr.ndim == 1:
            subarrays.append(ndim_1_to_str(subarr))
        else:
            subarrays.append(ndim_n_to_string(subarr))

    sep = os.linesep * (arr.ndim - 1)
    return sep.join(subarrays)
