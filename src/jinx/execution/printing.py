"""Methods for printing nouns (arrays, atoms)."""

import os

import numpy as np

from jinx.vocabulary import Atom, Array
from jinx.execution.conversion import is_box


MAX_COLS = 100


def noun_to_string(array: Atom | Array, max_cols: int = MAX_COLS) -> str:
    # TODO: Print boxes like J prints boxes.
    if is_box(array.implementation):
        return f"Box({array.implementation})"

    arr = np.atleast_1d(array.implementation)

    if arr.shape[-1] > max_cols:
        arr = arr[..., :max_cols]
        append_ellipsis = True
    else:
        append_ellipsis = False

    if np.issubdtype(arr.dtype, np.floating):
        rounded = [format_float(n) for n in arr.ravel().tolist()]
        arr = np.asarray(rounded).reshape(arr.shape)

    if np.issubdtype(arr.dtype, np.bool_):
        arr = arr.view(np.int8)

    if np.issubdtype(arr.dtype, np.str_):
        width = arr.shape[-1]
        arr = arr.view(f"<U{width}")

    arr_str = arr.astype(str)
    arr_str = np.strings.replace(arr_str, "-", "_")
    lengths = np.strings.str_len(arr_str)
    justify = np.max(lengths, axis=tuple(range(arr.ndim - 1)))
    arr_str = np.strings.rjust(arr_str, justify)
    return ndim_n_to_string(arr_str, append_ellipsis=append_ellipsis)


def get_decimal_places(n: float) -> int:
    n = abs(n)
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
    if n.is_integer():
        return f"{int(n)}"
    decimal_places = get_decimal_places(n)
    rounded_n = round(n, decimal_places)
    return f"{rounded_n}"


def ndim_1_to_str(arr: np.ndarray, append_ellipsis: bool) -> str:
    result = " ".join(arr.tolist())
    if append_ellipsis:
        result += " ..."
    return result


def ndim_n_to_string(arr: np.ndarray, append_ellipsis: bool) -> str:
    if arr.ndim == 1:
        return ndim_1_to_str(arr, append_ellipsis)

    subarrays = []
    for subarr in arr:
        if subarr.ndim == 1:
            subarrays.append(ndim_1_to_str(subarr, append_ellipsis))
        else:
            subarrays.append(ndim_n_to_string(subarr, append_ellipsis))

    sep = os.linesep * (arr.ndim - 1)
    return sep.join(subarrays)


def infer_print_height(array: np.ndarray) -> int:
    """Infer the height of the printed array."""
    if array.ndim <= 1:
        return 1

    shape = list(array.shape[:-1])
    height = 1
    sep = 0

    while shape:
        dim = shape.pop()
        height *= dim
        height += (dim - 1) * sep
        sep += 1

    return height


BOX_TOP_LEFT = "┌"
BOX_TOP_RIGHT = "┐"
BOX_BOTTOM_LEFT = "└"
BOX_BOTTOM_RIGHT = "┘"
BOX_HORIZONTAL = "─"
BOX_VERTICAL = "│"
BOX_T = "┬"
BOX_T_90 = "┤"
BOX_T_180 = "┴"
BOX_T_270 = "├"
BOX_CROSS = "┼"
