"""Methods for printing nouns (arrays, atoms)."""

import itertools
import os

import numpy as np

from jinx.vocabulary import Noun
from jinx.execution.conversion import is_box


MAX_COLS = 100


def noun_to_string(noun: Noun, max_cols: int = MAX_COLS) -> str:
    """Convert a noun to a string representation."""
    arr = noun.implementation
    rows = array_to_rows(arr, max_cols=max_cols)
    return os.linesep.join(rows)


def array_to_rows(arr: np.ndarray, max_cols: int = MAX_COLS) -> list[str]:
    """Convert an array to a list of strings for printing."""
    arr = np.atleast_1d(arr)

    if is_box(arr):
        return box_to_rows(arr)

    if arr.shape[-1] > max_cols:
        arr = arr[..., :max_cols]
        append_ellipsis = True
    else:
        append_ellipsis = False

    if np.issubdtype(arr.dtype, np.floating):
        rounded = [format_float(n) for n in arr.ravel().tolist()]
        arr = np.asarray(rounded).reshape(arr.shape)

    elif np.issubdtype(arr.dtype, np.bool_):
        arr = arr.view(np.int8)

    elif (
        np.issubdtype(arr.dtype, np.str_) and arr.dtype.itemsize == arr.dtype.alignment
    ):
        width = arr.shape[-1]
        arr = arr.view(f"{arr.dtype.byteorder}{arr.dtype.kind}{width}")

    arr_str = arr.astype(str)

    if np.issubdtype(arr.dtype, np.number):
        arr_str = np.strings.replace(arr_str, "-", "_")

    lengths = np.strings.str_len(arr_str)
    justify = np.max(lengths, axis=tuple(range(arr.ndim - 1)))
    arr_str = np.strings.rjust(arr_str, justify)
    return ndim_n_to_rows(arr_str, append_ellipsis=append_ellipsis)


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


def ndim_n_to_rows(arr: np.ndarray, append_ellipsis: bool) -> list[str]:
    if arr.ndim == 1:
        return [ndim_1_to_str(arr, append_ellipsis)]

    rows = []
    for n, item in enumerate(arr):
        if item.ndim == 1:
            rows.append(ndim_1_to_str(item, append_ellipsis))
        else:
            rows.extend(ndim_n_to_rows(item, append_ellipsis))
            if n < len(arr) - 1:
                rows.extend([""] * (arr.ndim - 2))
    return rows


def box_1D_or_2D_to_rows(box: list[list[str]], widths: list[int]) -> list[str]:
    """Convert a 2D box (list of list of strings) to a list of strings for printing."""
    rows = [box_top_line(widths=widths)]
    for n, box_row in enumerate(box):
        for row_item_row in itertools.zip_longest(*box_row, fillvalue=""):
            vals = [val.ljust(width) for val, width in zip(row_item_row, widths)]
            row = "│" + "│".join(vals) + "│"
            rows.append(row)

        if n < len(box) - 1:
            rows.append(box_row_divider_line(widths=widths))

    rows.append(box_bottom_line(widths=widths))
    return rows


def box_to_rows(box: np.ndarray) -> list[str]:
    """Convert a box to a list of strings for printing."""
    box_items = [item[0] for item in box.ravel()]
    items_as_rows = [array_to_rows(item) for item in box_items]

    row_groups = list(itertools.batched(items_as_rows, box.shape[-1], strict=True))
    if box.ndim >= 2:
        row_groups = list(itertools.batched(row_groups, box.shape[-2], strict=True))

    item_widths = np.array([len(rows[0]) for rows in items_as_rows]).reshape(box.shape)
    widths = np.max(item_widths, axis=tuple(range(box.ndim - 1))).tolist()

    if box.ndim == 1:
        return box_1D_or_2D_to_rows(row_groups, widths)

    if box.ndim == 2:
        return box_1D_or_2D_to_rows(row_groups[0], widths)

    boxes = [box_1D_or_2D_to_rows(rows, widths) for rows in row_groups]
    leading_dims = list(box.shape[:-2])

    while leading_dims:
        boxes = list(itertools.batched(boxes, leading_dims.pop(), strict=True))

    rows = []

    def flatten_to_rows(box_list, gap_size):
        for n, bxs in enumerate(box_list):
            if isinstance(bxs, str):
                rows.append(bxs)
            else:
                flatten_to_rows(bxs, gap_size - 1)
            if n < len(box_list) - 1:
                rows.extend([""] * gap_size)

    flatten_to_rows(boxes[0], box.ndim - 2)
    return rows


def box_top_line(widths: list[int]) -> str:
    return "┌" + "┬".join(["─" * width for width in widths]) + "┐"


def box_row_divider_line(widths: list[int]) -> str:
    return "├" + "┼".join(["─" * width for width in widths]) + "┤"


def box_bottom_line(widths: list[int]) -> str:
    return "└" + "┴".join(["─" * width for width in widths]) + "┘"
