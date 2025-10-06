import jax
from jinx.errors import JinxNotImplementedError
from jinx.execution.executor import Executor
from jinx.execution.jax.adverbs import ADVERB_MAP
from jinx.execution.jax.application import (
    apply_adverb,
    #     apply_conjunction,
    apply_dyad,
    apply_monad,
    build_fork,
    #     build_hook,
    ensure_noun_implementation,
)
from jinx.execution.jax.verbs import VERB_MAP

# from jinx.execution.numpy.conjunctions import CONJUNCTION_MAP
# from jinx.execution.numpy.conversion import ensure_noun_implementation
from jinx.execution.numpy.printing import noun_to_string

jax.config.update("jax_dynamic_shapes", True)


def make_not_implemented(name: str):
    def _not_implemented(*_, **__):
        raise JinxNotImplementedError(
            f"{name}: not yet implemented in the JAX executor."
        )

    return _not_implemented


executor = Executor[jax.Array](
    apply_monad=apply_monad,
    apply_dyad=apply_dyad,
    apply_conjunction=make_not_implemented("conjunction"),
    apply_adverb=apply_adverb,
    build_fork=build_fork,
    build_hook=make_not_implemented("hook"),
    ensure_noun_implementation=ensure_noun_implementation,
    primitive_verb_map=VERB_MAP,
    primitive_adverb_map=ADVERB_MAP,
    primitive_conjuction_map={},
    # Just use the NumPy implementation of printing.
    noun_to_string=noun_to_string,  # type: ignore[arg-type]
)
