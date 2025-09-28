"""Parsing and evaluation.

In J parsing and evaluation happen simultaneously. A fragment of the sentence is matched
against a set of 8 patterns. When a match is found the corresponding operation is executed
and the fragment is replaced with the result. This continues until no more matches are found.

In Jinx the execution may be done using different backends. An Executor instance with methods
for executing on each pattern is passed by the caller.

https://www.jsoftware.com/ioj/iojSent.htm#Parsing
https://www.jsoftware.com/help/jforc/parsing_and_execution_ii.htm
https://code.jsoftware.com/wiki/Vocabulary/Modifiers

"""

from typing import cast

from jinx.errors import EvaluationError, JinxNotImplementedError, JSyntaxError
from jinx.execution.executor import Executor
from jinx.primitives import PRIMITIVES
from jinx.vocabulary import (
    Adverb,
    Comment,
    Conjunction,
    Copula,
    Name,
    Noun,
    PartOfSpeechT,
    Punctuation,
    Verb,
)
from jinx.word_formation import form_words
from jinx.word_spelling import spell_words


def str_(executor: Executor, word: PartOfSpeechT | str) -> str:
    if isinstance(word, str):
        return word
    if isinstance(word, Noun):
        return executor.noun_to_string(word)
    elif isinstance(word, Verb | Adverb | Conjunction):
        return word.spelling
    elif isinstance(word, Name):
        return word.spelling
    elif isinstance(word, Punctuation | Copula):
        return word.spelling
    else:
        raise NotImplementedError(f"Cannot print word of type {type(word)}")


def print_words(
    executor: Executor, word: PartOfSpeechT, variables: dict[str, PartOfSpeechT]
) -> None:
    value = (
        str_(executor, variables[word.spelling])
        if isinstance(word, Name)
        else str_(executor, word)
    )
    print(value)


def evaluate_single_verb_sentence(
    executor: Executor, sentence: str, variables: dict[str, PartOfSpeechT]
) -> Verb:
    tokens = form_words(sentence)
    words = spell_words(tokens)
    result = _evaluate_words(executor, words, variables)
    if not isinstance(result, Verb):
        raise EvaluationError(f"Expected a verb, got {type(result).__name__}")
    return result


def build_verb_noun_phrase(
    executor: Executor,
    words: list[Verb | Noun | Adverb | Conjunction],
) -> Verb | Noun | None:
    """Build the verb or noun phrase from a list of words, or raise an error."""
    while len(words) > 1:
        match words:
            case [left, Adverb(), *remaining]:
                result = executor.apply_adverb(left, words[1])  # type: ignore[arg-type]
                words = [result, *remaining]

            case [left, Conjunction(), right, *remaining]:
                result = executor.apply_conjunction(left, words[1], right)  # type: ignore[arg-type]
                words = [result, *remaining]

            case _:
                raise EvaluationError("Unable to build verb/noun phrase")

    if not words:
        return None

    if isinstance(words[0], Verb | Noun):
        return words[0]

    raise EvaluationError("Unable to build verb/noun phrase")


def evaluate_words(
    executor: Executor,
    words: list[PartOfSpeechT],
    variables: dict[str, PartOfSpeechT] | None = None,
    level: int = 0,
) -> PartOfSpeechT:
    if variables is None:
        variables = {}

    # Ensure noun and verb implementations are set according to the chosen execution
    # framework (this is just NumPy for now).
    for word in words:
        if isinstance(word, Noun):
            executor.ensure_noun_implementation(word)

    all_primitives = (
        executor.primitive_verb_map
        | executor.primitive_adverb_map
        | executor.primitive_conjuction_map
    )

    for primitive in PRIMITIVES:
        if primitive.name not in all_primitives:
            continue
        if isinstance(primitive, Verb):
            monad, dyad = executor.primitive_verb_map[primitive.name]
            if primitive.monad is not None and monad is not None:
                primitive.monad.function = monad
            if primitive.dyad is not None and dyad is not None:
                primitive.dyad.function = dyad
        if isinstance(primitive, Adverb):
            primitive.function = executor.primitive_adverb_map[primitive.name]  # type: ignore[assignment]
        if isinstance(primitive, Conjunction):
            primitive.function = executor.primitive_conjuction_map[primitive.name]

    # Verb obverses are converted from strings to Verb objects.
    for word in words:
        if isinstance(word, Verb) and isinstance(word.obverse, str):
            verb = evaluate_single_verb_sentence(executor, word.obverse, variables)
            word.obverse = verb

    return _evaluate_words(executor, words, variables, level=level)


def get_parts_to_left(
    executor: Executor,
    word: PartOfSpeechT,
    words: list[PartOfSpeechT],
    current_level: int,
    variables: dict[str, PartOfSpeechT],
) -> list[Noun | Verb | Adverb | Conjunction]:
    """Get the parts of speach to the left of the current word, modifying list of remaining words.

    This method is called when the last word we encountered is an adverb or conjunction and
    a verb or noun phrase is expected to the left of it.
    """
    parts_to_left: list[Noun | Verb | Adverb | Conjunction] = []

    while words:
        word = resolve_word(word, variables)
        # A verb/noun phrase starts with a verb/noun which does not have a conjunction to its left.
        if isinstance(word, Noun | Verb):
            if not isinstance(words[-1], Conjunction):
                parts_to_left = [word, *parts_to_left]
                break
            else:
                conjunction = cast(Conjunction, words.pop())
                parts_to_left = [conjunction, word, *parts_to_left]

        elif isinstance(word, Adverb | Conjunction):
            parts_to_left = [word, *parts_to_left]

        elif isinstance(word, Punctuation) and word.spelling == ")":
            word = evaluate_words(executor, words, level=current_level + 1)
            continue

        else:
            break

        if words:
            word = words.pop()

    return parts_to_left


def resolve_word(
    word: PartOfSpeechT, variables: dict[str, PartOfSpeechT]
) -> PartOfSpeechT:
    """Find the Verb/Adverb/Conjunction/Noun that a name is assigned to.

    If we encounter a cycle of names, return the original name.
    """
    if not isinstance(word, Name):
        return word

    original_name = word
    visited = set()
    while True:
        visited.add(word.spelling)
        if word.spelling not in variables:
            return word
        assignment = variables[word.spelling]
        if not isinstance(assignment, Name):
            return assignment
        word = assignment
        if word.spelling in visited:
            return original_name


def _evaluate_words(
    executor: Executor, words: list[PartOfSpeechT], variables, level: int = 0
) -> PartOfSpeechT:
    # If the first word is None, prepend a None to the list denoting the left-most
    # edge of the expression.
    if words[0] is not None:
        words = [None, *words]  # type: ignore[list-item]

    fragment: list[PartOfSpeechT] = []
    result: PartOfSpeechT

    while words:
        word = words.pop()

        if isinstance(word, Comment):
            continue

        elif isinstance(word, (Punctuation, Copula)):
            word = word.spelling  # type: ignore[assignment]

        # If the next word closes a parenthesis, we need to evaluate the words inside it
        # first to get the next word to prepend to the fragment.
        if word == ")":
            word = evaluate_words(executor, words, variables, level=level + 1)

        # If the fragment has a modifier (adverb/conjunction) at the start, we need to find the
        # entire verb/noun phrase to the left as the next word to prepend to the fragment.
        # Contrary to usual parsing and evaluation, the verb/noun phrase is evaluated left-to-right.
        if fragment and isinstance(
            resolve_word(fragment[0], variables), Adverb | Conjunction
        ):
            parts_to_left = get_parts_to_left(
                executor, word, words, level, variables=variables
            )
            word = build_verb_noun_phrase(executor, parts_to_left)  # type: ignore[assignment]

        fragment = [word, *fragment]

        # fmt: off
        while True:

            # 7. Is
            # This case (assignment) is checked separately, before names are substituted with their values.
            # For now we treat =. and =: the same.
            match fragment:
                case Name(), "=." | "=:", Conjunction() | Adverb() | Verb() | Noun() | Name(), *_:
                    name, _, cavn, *last = fragment
                    name = cast(Name, name)
                    variables[name.spelling] = cavn
                    fragment = [name, *last]
                    continue

            # Substitute variable names with their values and do pattern matching. If a match occurs
            # the original fragment (list of unsubstituted names) is modified.
            fragment_ = [resolve_word(word, variables) for word in fragment]

            match fragment_:

                # 0. Monad
                case None | "=." | "=:" | "(", Verb(), Noun():
                    edge, verb, noun = fragment_
                    result = executor.apply_monad(verb, noun) # type: ignore[arg-type]
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 1. Monad
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Adverb() | Verb(), Verb(), Noun():
                    edge, _, verb, noun = fragment_
                    result = executor.apply_monad(verb, noun) # type: ignore[arg-type]
                    fragment[2:] = [result]

                # 2. Dyad
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Noun(), Verb(), Noun():
                    edge, noun, verb, noun_2 = fragment_
                    result = executor.apply_dyad(verb, noun, noun_2) # type: ignore[arg-type]
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 3. Adverb
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Adverb(),
                    *_,
                ):
                    edge, verb, adverb, *last = fragment_
                    result = executor.apply_adverb(verb, adverb) # type: ignore[arg-type]
                    if edge == "(" and last == [")"] and level > 0:
                        return result
                    fragment[1:3] = [result]

                # 4. Conjunction
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Conjunction(),
                    Verb() | Noun(),
                    *_,
                ):
                    edge, verb_or_noun_1, conjunction, verb_or_noun_2, *last = fragment_
                    result = executor.apply_conjunction(verb_or_noun_1, conjunction, verb_or_noun_2) # type: ignore[arg-type]
                    if edge == "(" and last == [")"] and level > 0:
                        return result
                    fragment[1:4] = [result]

                # 5. Fork
                case (
                    None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                    Verb() | Noun(),
                    Verb(),
                    Verb(),
                ):
                    edge, verb_or_noun_1, verb_2, verb_3 = fragment_
                    result = executor.build_fork(verb_or_noun_1, verb_2, verb_3) # type: ignore[arg-type]
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:4] = [result]

                # 6. Hook/Adverb
                case (
                    None | "=." | "=:" | "(",
                    Conjunction() | Adverb() | Verb() | Noun(),
                    Conjunction() | Adverb() | Verb() | Noun(),
                    *_,
                ):
                    edge, cavn1, cavn2, *last = fragment_
                    match [cavn1, cavn2]:
                        case [Verb(), Verb()]:
                            result = executor.build_hook(cavn1, cavn2) # type: ignore[arg-type]
                        case [Adverb(), Adverb()] | [Conjunction() , Noun()] | [Conjunction(), Verb()] | [Noun(), Conjunction()] | [Verb(), Conjunction()]:
                            # These are valid combinations but not implemented yet.
                            raise JinxNotImplementedError(
                                f"Jinx error: currently only 'Verb Verb' is implemented for hook/adverb matching, got "
                                f"({type(cavn1).__name__} {type(cavn2).__name__})"
                            )
                        case _:
                            raise JSyntaxError(f"syntax error: unexecutable fragment ({type(cavn1).__name__} {type(cavn2).__name__})")
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 8. Parentheses
                # Differs from the J source as it does not match ")" and instead checks
                # the level to ensure that "(" is balanced.
                case ["(", Conjunction() | Adverb() | Verb() | Noun()]:
                    _, cavn = fragment_
                    if level > 0:
                        return cast(PartOfSpeechT, cavn)
                    raise EvaluationError("Unbalanced parentheses")

                # Non-executable fragment.
                case _:
                    break

        # fmt: on

    if len(fragment) > 2:
        raise EvaluationError(
            f"Unexecutable fragment: {[str_(executor, w) for w in fragment if w is not None]}"
        )

    return fragment[1]
