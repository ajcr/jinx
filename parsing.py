"""Parsing and evaluation.

In J, parsing and evaluation happen simultaneously.

https://www.jsoftware.com/ioj/iojSent.htm#Parsing
https://www.jsoftware.com/help/jforc/parsing_and_execution_ii.htm

"""

from vocabulary import (
    PartOfSpeechT,
    Verb,
    Adverb,
    Conjunction,
    Punctuation,
    Noun,
    Name,
    Comment,
    Copula,
)


def evaluate(words: list[PartOfSpeechT]) -> list[PartOfSpeechT]:
    """Evaluate the words in the sentence."""

    fragment = []

    while words:
        word = words.pop()

        if isinstance(word, Comment):
            continue

        elif isinstance(word, (Punctuation, Copula)):
            word = word.spelling

        fragment = [word, *fragment]

        match fragment:

            # 0. Monad
            case "=." | "=:" | "(", Verb(), Noun(), _:
                pass

            # 1. Monad
            case "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Verb(), Verb(), Noun():
                pass

            # 2. Dyad
            case "=." | "=:" | "(" | Adverb() | Verb() | Noun(), Noun(), Verb(), Noun():
                pass

            # 3. Adverb
            case (
                "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                Verb() | Noun(),
                Adverb(),
                _,
            ):
                pass

            # 4. Conjunction
            case (
                "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                Verb() | Noun(),
                Conjunction(),
                Verb() | Noun(),
            ):
                pass

            # 5. Fork
            case (
                "=." | "=:" | "(" | Adverb() | Verb() | Noun(),
                Verb() | Noun(),
                Verb(),
                Verb(),
            ):
                pass

            # 6. Hook/Adverb
            case (
                "=." | "=:" | "(",
                Conjunction() | Adverb() | Verb() | Noun(),
                Conjunction() | Adverb() | Verb() | Noun(),
                _,
            ):
                pass

            # 7. Is
            case Name(), "=." | "=:", Conjunction() | Adverb() | Verb() | Noun(), _:
                pass

            # 8. Parentheses
            case "(", Conjunction() | Adverb() | Verb() | Noun(), ")", _:
                pass

            # No default case.

    return fragment
