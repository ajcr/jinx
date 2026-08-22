import numpy as np
from jinx.execution.executor import Executor
from jinx.execution.numpy.adverbs import ADVERB_MAP
from jinx.execution.numpy.application import (
    apply_adverb,
    apply_conjunction,
    apply_dyad,
    apply_monad,
)
from jinx.execution.numpy.conjunctions import CONJUNCTION_MAP
from jinx.execution.numpy.conversion import (
    convert_python_object_to_noun,
    ensure_noun_implementation,
    to_verb_str,
)
from jinx.execution.numpy.printing import noun_to_string
from jinx.execution.numpy.trains import build_fork, build_hook
from jinx.execution.numpy.verbs import VERB_MAP
from jinx.vocabulary import get_spelling_for_noun_as_part_of_verb

# Register method for spelling an np.ndarray as part of another entity.
get_spelling_for_noun_as_part_of_verb.register(to_verb_str)


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
    python_object_to_noun=convert_python_object_to_noun,
)
