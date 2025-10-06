from typing import Callable

import jax
import jax.numpy as jnp

MonadT = Callable[[jax.Array], jax.Array]
DyadT = Callable[[jax.Array, jax.Array], jax.Array]


@jax.jit
def number_monad(y: jax.Array) -> jax.Array:
    """# monad: count number of items in y."""
    if jnp.isscalar(y) or y.shape == ():
        return jnp.array(1)
    return jnp.array(y.shape[0])


@jax.jit
def percentco_dyad(x: jax.Array, y: jax.Array) -> jax.Array:
    return jnp.power(y, 1 / x)


@jax.jit
def plusdot_monad(y: jax.Array) -> jax.Array:
    """+. monad: returns real and imaginary parts of numbers."""
    return jnp.stack([jnp.real(y), jnp.imag(y)], axis=-1)


@jax.jit
def plusco_monad(y: jax.Array) -> jax.Array:
    """+: monad: double the values in the array."""
    return 2 * y


@jax.jit
def minusdot_monad(y: jax.Array) -> jax.Array:
    """-.: monad: returns 1 - y."""
    return 1 - y


@jax.jit
def minusco_monad(y: jax.Array) -> jax.Array:
    """-: monad: halve the values in the array."""
    return y / 2


@jax.jit
def minusco_dyad(x: jax.Array, y: jax.Array) -> jax.Array:
    """-: dyad: match, returns true if x and y have same shape and values."""
    is_equal = jnp.array_equal(x, y, equal_nan=True)
    return jnp.asarray(is_equal)


@jax.jit
def hatdot_dyad(x: jax.Array, y: jax.Array) -> jax.Array:
    """^. dyad: logarithm of y to the base x."""
    return jnp.log(y) / jnp.log(x)


@jax.jit
def ltco_monad(y: jax.Array) -> jax.Array:
    """<: monad: decrements the array."""
    return y - 1


@jax.jit
def gtco_monad(y: jax.Array) -> jax.Array:
    """>: monad: increments the array."""
    return y + 1


VERB_MAP: dict[str, tuple[MonadT | None, DyadT | None]] = {
    # VERB: (MONAD, DYAD)
    "PLUS": (jnp.conj, jnp.add),
    "PLUSDOT": (plusdot_monad, NotImplemented),
    "PLUSCO": (plusco_monad, NotImplemented),
    "MINUS": (jnp.negative, jnp.subtract),
    "MINUSDOT": (minusdot_monad, NotImplemented),
    "MINUSCO": (minusco_monad, minusco_dyad),
    "STAR": (jnp.sign, jnp.multiply),
    "PERCENT": (jnp.reciprocal, jnp.divide),
    "HAT": (jnp.exp, jnp.power),
    "HATDOT": (jnp.log, hatdot_dyad),
    "LTDOT": (jnp.floor, jnp.minimum),
    "GTDOT": (jnp.ceil, jnp.maximum),
    "LTCO": (ltco_monad, jnp.less_equal),
    "GTCO": (gtco_monad, jnp.greater_equal),
    "NUMBER": (number_monad, NotImplemented),
    "BAR": (jnp.abs, NotImplemented),
}
