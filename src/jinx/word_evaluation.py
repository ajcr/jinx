"""Parsing and evaluation.

In J, parsing and evaluation happen simultaneously.

https://www.jsoftware.com/ioj/iojSent.htm#Parsing
https://www.jsoftware.com/help/jforc/parsing_and_execution_ii.htm
https://code.jsoftware.com/wiki/Vocabulary/Modifiers

"""

from jinx.vocabulary import (
    PartOfSpeechT,
    Verb,
    Adverb,
    Conjunction,
    Punctuation,
    Noun,
    Name,
    Comment,
    Copula,
    Atom,
    Array,
)

from jinx.execution.application import (
    apply_monad,
    apply_dyad,
    apply_conjunction,
    apply_adverb,
    build_fork,
    build_hook,
    ensure_verb_implementation,
)
from jinx.execution.conversion import ensure_noun_implementation

from jinx.execution.printing import (
    atom_to_string,
    array_to_string,
)


class EvaluationError(Exception):
    pass


def str_(word: Atom | Array | Verb | Conjunction | Adverb) -> str:
    if isinstance(word, str):
        return word
    if isinstance(word, Atom):
        return atom_to_string(word)
    elif isinstance(word, Array):
        return array_to_string(word)
    elif isinstance(word, (Verb, Conjunction)):
        return word.spelling
    elif isinstance(word, Adverb):
        return word.spelling
    else:
        raise NotImplementedError(f"Cannot print word of type {type(word)}")


def print_words(words: list[PartOfSpeechT]) -> None:
    print(" ".join(str_(word) for word in words if word is not None))


def build_verb_noun_phrase(
    words: list[Verb | Noun | Adverb | Conjunction],
) -> Verb | Noun:
    assert len(words) > 0
    if len(words) == 1:
        return words[0]
    words = words.copy()
    while len(words) > 1:
        if isinstance(words[1], Adverb):
            result = apply_adverb(words.pop(0), words.pop(0))
            words = [result, *words]
        elif isinstance(words[1], Conjunction):
            result = apply_conjunction(words.pop(0), words.pop(0), words.pop(0))
            words = [result, *words]
        else:
            raise EvaluationError("Unable to build verb/noun phrase")
    return result


def evaluate_words(words: list[PartOfSpeechT], level: int = 0) -> list[PartOfSpeechT]:
    # Ensure noun and verb implementations are set according to the chosen execution
    # framework (this is just NumPy for now).
    for word in words:
        if isinstance(word, Noun):
            ensure_noun_implementation(word)
        elif isinstance(word, Verb):
            ensure_verb_implementation(word)

    # If the first word is None, prepend a None to the list denoting the left-most
    # edge of the expression.
    if words[0] is not None:
        words = [None, *words]

    fragment = []

    while words:
        word = words.pop()

        if isinstance(word, Comment):
            continue

        elif isinstance(word, (Punctuation, Copula)):
            word = word.spelling

        # TODO: If word is a Name, lookup the Verb/Noun it refers to and add that.

        # If the next word closes a parenthesis, we need to evaluate the words inside it
        # first to get the next word to prepend to the fragment.
        if word == ")":
            word = evaluate_words(words, level=level + 1)

        # If the fragment has a modifier (adverb/conjunction) at the start, we need to find the
        # entire verb/noun phrase to the left as the next word to prepend to the fragment.
        # Contrary to usual parsing and evaluation, the verb/noun phrase is evaluated left-to-right.
        # fmt: off
        if fragment and isinstance(fragment[0], Adverb | Conjunction):
            parts_to_left = []

            while words:
                # A verb/noun phrase starts with a verb/noun which does not have a conjunction to its left.
                if isinstance(word, Noun | Verb):
                    if not isinstance(words[-1], Conjunction):
                        parts_to_left = [word, *parts_to_left]
                        break
                    else:
                        conjunction = words.pop()
                        parts_to_left = [conjunction, word, *parts_to_left]

                elif isinstance(word, Adverb | Conjunction):
                    parts_to_left = [word, *parts_to_left]

                elif word == ")":
                    word = evaluate_words(words, level=level + 1)
                    continue

                else:
                    break

                if words:
                    word = words.pop()

            # evaluate the parts_to_left sequence (return single noun/verb)
            if not parts_to_left:
                continue
            word = build_verb_noun_phrase(parts_to_left)

        fragment = [word, *fragment]

        while True:
            match fragment:

                # 0. Monad
                case None | "=." | "=:" | "(", Verb(), Noun():
                    edge, verb, noun = fragment
                    result = apply_monad(verb, noun)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 1. Monad
                # N.B. differs from reference in that the 'edge' can be a conjunction.
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun() | Conjunction(), Adverb() | Verb(), Verb(), Noun():
                    edge, _, verb, noun = fragment
                    result = apply_monad(verb, noun)
                    fragment[2:] = [result]

                # Another monad case for a conjunction that is not in the J reference.
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Conjunction(), Noun(), Verb(), Noun():
                    edge, conjunction, noun, verb, noun_2 = fragment
                    result = apply_monad(verb, noun_2)
                    fragment[3:] = [result]

                # 2. Dyad
                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Noun(), Verb(), Noun():
                    edge, noun, verb, noun_2 = fragment
                    result = apply_dyad(verb, noun, noun_2)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 3. Adverb
                case (
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb(), Adverb()] | 
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb(), Adverb(), *_]
                ):
                    edge, verb, adverb, *last = fragment

                    if isinstance(edge, Adverb):
                        # TODO: grab entire verb/noun phrase to the left of the adverb
                        raise NotImplementedError("adverb application to adverb")

                    else:
                        result = apply_adverb(verb, adverb)

                    if edge == "(" and last == [")"] and level > 0:
                        return result

                    fragment[1:3] = [result]

                case None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Noun(), Adverb():
                    edge, noun, adverb = fragment
                    raise NotImplementedError("adverb application to noun not implemented")

                # 4. Conjunction
                case (
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb() | Noun(), Conjunction(), Verb() | Noun()] |
                    [None | "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb() | Noun(), Conjunction(), Verb() | Noun(), _]
                ):
                    edge, verb_or_noun_1, conjunction, verb_or_noun_2, *last = fragment
                    # TODO: find entire verb phrase on the left of the conjunction before applying the conjunction
                    # See: https://code.jsoftware.com/wiki/Vocabulary/Modifiers
                    result = apply_conjunction(verb_or_noun_1, conjunction, verb_or_noun_2)
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
                    edge, verb_or_noun_1, verb_2, verb_3 = fragment
                    if not isinstance(verb_2, Verb):
                        raise NotImplementedError("Fork currently implemented only for verb/verb/verb")
                    result = build_fork(verb_or_noun_1, verb_2, verb_3)
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 6. Hook/Adverb
                case (
                    [None | "=." | "=:" | "(", Conjunction() | Adverb() | Verb() | Noun(), Conjunction() | Adverb() | Verb() | Noun()]
                ):
                    edge, cavn1, cavn2, *last = fragment
                    if isinstance(cavn1, Verb) and isinstance(cavn2, Verb):
                        result = build_hook(cavn1, cavn2)
                    else:
                        raise NotImplementedError("Only VV is implemented for hook/adverb matching")
                    if edge == "(" and level > 0:
                        return result
                    fragment[1:] = [result]

                # 7. Is
                case Name(), "=." | "=:", Conjunction() | Adverb() | Verb() | Noun(), _:
                    raise NotImplementedError("copula")

                # 8. Parentheses
                # Differs from the J source as it does not match ")" and instead checks
                # the level to ensure that "(" is balanced.
                case ["(", Conjunction() | Adverb() | Verb() | Noun()]:
                    _, cavn = fragment
                    if level > 0:
                        return cavn
                    raise EvaluationError("Unbalanced parentheses")

                # Non-executable fragment.
                case _:
                    break

        # fmt: on

    if len(fragment) > 2 and level > 0:
        raise EvaluationError("Unexecutable fragment")

    return fragment
