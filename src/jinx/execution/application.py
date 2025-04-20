"""Methods for applying verb implementations to nouns and verbs."""

import numpy as np

from jinx.vocabulary import Noun, Atom, Verb, Conjunction, Adverb, Monad
from jinx.execution.conversion import (
    ensure_noun_implementation,
    ndarray_or_scalar_to_noun,
    is_ufunc,
)
from jinx.execution.primitives import PRIMITIVE_MAP


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


def apply_monad(verb: Verb, noun: Noun) -> Noun:
    ensure_noun_implementation(noun)
    ensure_verb_implementation(verb)

    arr = noun.implementation

    if isinstance(noun, Atom):
        return ndarray_or_scalar_to_noun(verb.monad.function(arr))

    verb_rank = verb.monad.rank
    if verb_rank < 0:
        raise NotImplementedError("Negative verb rank not yet supported")

    noun_rank = noun.implementation.ndim
    r = min(verb_rank, noun_rank)

    # If the verb rank is 0 it applies to each atom of the array.
    # NumPy's unary ufuncs are typically designed to work this way
    # Apply the function directly here as an optimisation.
    if r == 0 and is_ufunc(verb.monad.function):
        return ndarray_or_scalar_to_noun(verb.monad.function(arr))

    # Look at the shape of the array and the rank of the verb to
    # determine the frame and cell shape.
    #
    # The trailing r axes define the cell shape and the preceding
    # axes define the frame shape. E.g. for r=2:
    #
    #   arr.shape = (n0, n1, n2, n3, n4)
    #                ----------  ------
    #                ^ frame     ^ cell
    #
    # If r=0, the frame shape is the same as the shape and the monad
    # applies to each atom of the array.
    if r == 0:
        frame_shape = arr.shape
        arr_reshaped = arr.ravel()
    else:
        cell_shape = arr.shape[-r:]
        frame_shape = arr.shape[:-r]
        arr_reshaped = arr.reshape(-1, *cell_shape)

    cells = [verb.monad.function(cell) for cell in arr_reshaped]
    cells = maybe_pad_with_fill_value(cells)
    result = np.asarray(cells).reshape(frame_shape + cells[0].shape)
    return ndarray_or_scalar_to_noun(result)


def apply_dyad(verb: Verb, noun_1: Noun, noun_2: Noun) -> Noun:
    # For now, just apply the verb with no regard to its rank.

    ensure_noun_implementation(noun_1)
    ensure_noun_implementation(noun_2)
    ensure_verb_implementation(verb)

    result = verb.dyad.function(noun_1.implementation, noun_2.implementation)
    return ndarray_or_scalar_to_noun(result)


# Applying a conjunction usually produces a verb (but not always).
# Assume that it's just a verb for now.
def apply_conjunction(
    verb_or_noun_1: Verb | Noun, conjunction: Conjunction, verb_or_noun_2: Verb | Noun
) -> Verb:
    if isinstance(verb_or_noun_1, Noun):
        ensure_noun_implementation(verb_or_noun_1)
    if isinstance(verb_or_noun_2, Noun):
        ensure_noun_implementation(verb_or_noun_2)
    if isinstance(verb_or_noun_1, Verb):
        ensure_verb_implementation(verb_or_noun_1)
    if isinstance(verb_or_noun_2, Verb):
        ensure_verb_implementation(verb_or_noun_2)

    f = PRIMITIVE_MAP[conjunction.name]
    return f(verb_or_noun_1, verb_or_noun_2)


def apply_adverb_to_verb(verb: Verb, adverb: Adverb) -> Verb:
    if adverb.name in PRIMITIVE_MAP:
        monad_function = PRIMITIVE_MAP[adverb.name][0](verb)
    else:
        raise NotImplementedError(f"Adverb '{adverb.spelling}' not supported")

    spelling = verb.spelling + adverb.spelling

    return Verb(
        spelling=spelling,
        name=spelling,
        monad=Monad(name=spelling, rank=float("inf"), function=monad_function),
    )


def ensure_verb_implementation(verb: Verb) -> None:
    if verb.monad and verb.monad.function is None and verb.name in PRIMITIVE_MAP:
        verb.monad.function = PRIMITIVE_MAP[verb.name][0]
    if verb.dyad and verb.dyad.function is None and verb.name in PRIMITIVE_MAP:
        verb.dyad.function = PRIMITIVE_MAP[verb.name][1]
