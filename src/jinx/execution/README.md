# Execution

Jinx supports alternative backends for executing J sentences.

The following execution backends are available.

## [NumPy](https://numpy.org/)

The default Jinx executor, implements many J primitives.

## [JAX](https://docs.jax.dev/en/latest/index.html)

Highly experimental. Most functionality is not present, or else likely to be incomplete.

The emphesis is on creating JIT-compilable and composable primitives, which may often be at odds with J's dynamic nature and is likely to restrict how much of the language can be implemented.

Currently only CPU backends are targeted, but support for accelerators could be included in future.
