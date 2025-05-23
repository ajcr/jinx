"""Methods for printing nouns (arrays, atoms)."""

import os

import numpy as np

from jinx.vocabulary import Atom, Array


def format_numeric(n):
    if n < 0:
        return f"_{-n}"
    return f" {n}"


def atom_to_string(atom: Atom) -> str:
    n = atom.implementation
    if np.isinf(n):
        return "__" if n < 0 else "_"
    if n < 0:
        return f"_{-n}"
    return str(n)


def array_to_string_float(array: Array) -> str:
    np.set_printoptions(
        infstr="_", formatter={"int_kind": format_numeric, "float_kind": format_numeric}
    )
    return str(array.implementation)


def rank_1_array_float_to_string(
    arr: np.ndarray, justify: list[int], append_ellisis: bool = False
) -> list[int]:
    assert arr.ndim == 1
    assert len(justify) == len(arr)

    arr_python = arr.tolist()
    scalars = []

    for scalar, pad in zip(arr_python, justify):
        if np.isinf(scalar):
            str_ = scalar = "_" if scalar < 0 else "__"
        else:
            if scalar.is_integer():
                scalar = int(scalar)
            sign = "_" if scalar < 0 else ""
            str_ = f"{sign}{abs(scalar)}"
        str_ = str_.rjust(pad)
        scalars.append(str_)

    if append_ellisis:
        scalars.append("...")

    return scalars


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

    # If the array is a boolean array, convert it to int8 for printing.
    if np.issubdtype(dtype, np.bool_):
        arr = arr.view(np.int8)

    # For now, just use NumPy for printing float arrays
    if np.issubdtype(dtype, np.floating):
        return array_to_string_float(array)

    if arr.shape[-1] > max_cols:
        arr = arr[..., :max_cols]
        append_ellipsis = True
    else:
        append_ellipsis = False

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
