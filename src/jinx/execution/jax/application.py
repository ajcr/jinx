import functools
from dataclasses import dataclass

import jax
import jax.numpy as jnp
from jinx.errors import JinxNotImplementedError, ValenceError
from jinx.vocabulary import Adverb, DataType, Dyad, Monad, Noun, RankT, Verb

DATATYPE_TO_NP_MAP = {
    # JAX requires support for int64 to be set via config.
    DataType.Integer: jnp.int32,
    DataType.Float: jnp.float64,
}


INFINITY = float("inf")


def get_rank(verb_rank: RankT, noun_rank: int) -> int:
    """Get the rank at which to apply the verb to the noun.

    If the verb rank is negative, it means that the verb rank is subtracted
    from the noun rank, to a minimum of 0.
    """
    if verb_rank < 0:
        return max(0, noun_rank + verb_rank)  # type: ignore[return-value]
    return min(verb_rank, noun_rank)  # type: ignore[return-value]


@dataclass
class ArrayCells:
    cell_shape: tuple[int, ...]
    frame_shape: tuple[int, ...]
    cells: jax.Array


def split_into_cells(arr: jax.Array, rank: int) -> ArrayCells:
    """
    Look at the array shape and rank to determine frame and cell shape.

    The trailing `rank` axes define the cell shape and the preceding
    axes define the frame shape. E.g. for rank=2:

      arr.shape = (n0, n1, n2, n3, n4)
                   ----------  ------
                   ^ frame     ^ cell

    If rank=0, the frame shape is the same as the shape and the monad
    applies to each atom of the array.
    """
    if rank == 0:
        return ArrayCells(cell_shape=(), frame_shape=arr.shape, cells=arr.ravel())

    return ArrayCells(
        cell_shape=arr.shape[-rank:],
        frame_shape=arr.shape[:-rank],
        cells=arr.reshape(-1, *arr.shape[-rank:]),
    )


def infer_data_type(data: jax.Array) -> DataType:
    dtype = data.dtype
    if jnp.issubdtype(dtype, jnp.integer) or jnp.issubdtype(dtype, jnp.bool_):
        return DataType.Integer
    if jnp.issubdtype(dtype, jnp.floating):
        return DataType.Float

    raise NotImplementedError(f"Cannot handle JAX dtype: {dtype}")


def convert_noun_to_jax_array(noun: Noun[jax.Array]) -> jax.Array:
    dtype = DATATYPE_TO_NP_MAP[noun.data_type]
    if len(noun.data) == 1:
        # A scalar (ndim == 0) is returned for single element arrays.
        return jnp.array(noun.data[0], dtype=dtype)  # type: ignore[call-overload]
    return jnp.array(noun.data, dtype=dtype)  # type: ignore[call-overload]


def ensure_noun_implementation(noun: Noun[jax.Array]) -> None:
    if noun.implementation is None:
        noun.implementation = convert_noun_to_jax_array(noun)


def jax_array_to_noun(array: jax.Array) -> Noun[jax.Array]:
    data_type = infer_data_type(array)
    return Noun[jax.Array](data_type=data_type, implementation=array)


def apply_monad(verb: Verb[jax.Array], noun: Noun[jax.Array]) -> Noun[jax.Array]:
    result = _apply_monad(verb, noun.implementation)
    return jax_array_to_noun(result)


def _apply_monad(verb: Verb[jax.Array], arr: jax.Array) -> jax.Array:
    if verb.monad is None or verb.monad.function is None:
        raise ValenceError(f"Verb {verb.spelling} has no monadic valence.")
    if verb.monad.function is NotImplemented:
        raise JinxNotImplementedError(
            f"Verb {verb.spelling} monad function is not yet implemented in Jinx."
        )

    if isinstance(verb.monad.function, Verb):
        function = functools.partial(_apply_monad, verb.monad.function)
    else:
        function = verb.monad.function  # type: ignore[assignment]

    rank = get_rank(verb.monad.rank, arr.ndim)

    if rank == 0:
        return function(arr)

    array_cells = split_into_cells(arr, rank)

    if array_cells.cells.ndim == 1:
        cells = function(array_cells.cells)
    else:
        cells = jnp.apply_along_axis(function, axis=-1, arr=array_cells.cells)

    # No filling/padding for now...
    return jnp.asarray(cells).reshape(array_cells.frame_shape + cells[0].shape)


def apply_dyad(
    verb: Verb[jax.Array], noun_1: Noun[jax.Array], noun_2: Noun[jax.Array]
) -> Noun[jax.Array]:
    result = _apply_dyad(verb, noun_1.implementation, noun_2.implementation)
    return jax_array_to_noun(result)


def _apply_dyad(
    verb: Verb[jax.Array], left_arr: jax.Array, right_arr: jax.Array
) -> jax.Array:
    if verb.dyad is None or verb.dyad.function is None:
        raise ValenceError(f"Verb {verb.spelling} has no dyadic valence.")
    if verb.dyad.function is NotImplemented:
        raise JinxNotImplementedError(
            f"Verb {verb.spelling} dyad function is not yet implemented."
        )

    if isinstance(verb.dyad.function, Verb):
        function = functools.partial(_apply_dyad, verb.dyad.function)
    else:
        function = verb.dyad.function  # type: ignore[assignment]

    left_rank = get_rank(verb.dyad.left_rank, left_arr.ndim)
    right_rank = get_rank(verb.dyad.right_rank, right_arr.ndim)

    if left_rank == right_rank == 0 and (left_arr.ndim == 0 or right_arr.ndim == 0):
        return function(left_arr, right_arr)

    left = split_into_cells(left_arr, left_rank)
    right = split_into_cells(right_arr, right_rank)

    if left.frame_shape == right.frame_shape:
        cells = [
            function(left_cell, right_cell)
            for left_cell, right_cell in zip(left.cells, right.cells, strict=True)
        ]
        return jnp.asarray(cells).reshape(left.frame_shape + cells[0].shape)

    raise JinxNotImplementedError(
        "Dyadic verbs with non-zero rank and different frame shape are not yet implemented."
    )


def apply_adverb(verb_or_noun: Verb | Noun, adverb: Adverb) -> Verb:
    return adverb.function(verb_or_noun)


def build_fork(
    f: Verb[jax.Array] | Noun[jax.Array], g: Verb[jax.Array], h: Verb[jax.Array]
) -> Verb[jax.Array]:
    """Build a fork given verbs f, g, h.

      (f g h) y  ->    (f y) g   (h y)
    x (f g h) y  ->  (x f y) g (x h y)

    The new verb has infinite rank.

    Note that f can be a noun, in which case there is one fewer function calls.
    """

    @jax.jit
    def _monad(y: jax.Array) -> jax.Array:
        if isinstance(f, Verb) and f.spelling == "[:":
            hy = _apply_monad(h, y)
            return _apply_monad(g, hy)

        if isinstance(f, Verb):
            a = _apply_monad(f, y)
        else:
            a = f.implementation
        b = _apply_monad(h, y)
        return _apply_dyad(g, a, b)

    @jax.jit
    def _dyad(x: jax.Array, y: jax.Array) -> jax.Array:
        if isinstance(f, Verb) and f.spelling == "[:":
            hy = _apply_dyad(h, x, y)
            return _apply_monad(g, hy)

        if isinstance(f, Verb):
            a = _apply_dyad(f, x, y)
        else:
            a = f.implementation
        b = _apply_dyad(h, x, y)
        return _apply_dyad(g, a, b)

    if isinstance(f, Verb):
        f_spelling = f"({f.spelling})" if " " in f.spelling else f.spelling
    else:
        f_spelling = str(f.implementation)

    g_spelling = f"({g.spelling})" if " " in g.spelling else g.spelling
    h_spelling = f"({h.spelling})" if " " in h.spelling else h.spelling
    spelling = f"{f_spelling} {g_spelling} {h_spelling}"

    return Verb[jax.Array](
        spelling=spelling,
        name=spelling,
        monad=Monad(
            name=spelling,
            rank=INFINITY,
            function=_monad,
        ),
        dyad=Dyad(
            name=spelling,
            left_rank=INFINITY,
            right_rank=INFINITY,
            function=_dyad,
        ),
    )
