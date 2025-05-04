"""Helper methods for manipulating arrays."""

import numpy as np


def maybe_pad_with_fill_value(
    arrays: list[np.ndarray], fill_value: int = 0
) -> list[np.ndarray]:
    shapes = [arr.shape for arr in arrays]
    if len(set(shapes)) == 1:
        return arrays

    if len({len(shape) for shape in shapes}) != 1:
        raise NotImplementedError("Cannot pad arrays of different ranks")

    dims = [max(dim) for dim in zip(*shapes)]
    padded_arrays = []

    for arr in arrays:
        pad_widths = [(0, dim - shape) for shape, dim in zip(arr.shape, dims)]
        padded_array = np.pad(
            arr, pad_widths, mode="constant", constant_values=fill_value
        )
        padded_arrays.append(padded_array)

    return padded_arrays
