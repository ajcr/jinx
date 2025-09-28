import numpy as np
from jinx.execution.executor import Executor
from jinx.execution.numpy.adverbs import ADVERB_MAP
from jinx.execution.numpy.application import (
    apply_adverb,
    apply_conjunction,
    apply_dyad,
    apply_monad,
    build_fork,
    build_hook,
)
from jinx.execution.numpy.conjunctions import CONJUNCTION_MAP
from jinx.execution.numpy.conversion import ensure_noun_implementation
from jinx.execution.numpy.printing import noun_to_string
from jinx.execution.numpy.verbs import VERB_MAP

executor = Executor[np.ndarray](
    apply_monad=apply_monad,
    apply_dyad=apply_dyad,
    apply_conjunction=apply_conjunction,
    apply_adverb=apply_adverb,
    build_fork=build_fork,
    build_hook=build_hook,
    ensure_noun_implementation=ensure_noun_implementation,
    primitive_verb_map=VERB_MAP,
    primitive_adverb_map=ADVERB_MAP,
    primitive_conjuction_map=CONJUNCTION_MAP,  # type: ignore[arg-type]
    noun_to_string=noun_to_string,
)
