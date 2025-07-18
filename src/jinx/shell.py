import cmd
import sys

from jinx.errors import BaseJError, SpellingError
from jinx.word_formation import form_words
from jinx.word_spelling import spell_words
from jinx.word_evaluation import evaluate_words, print_words


class Shell(cmd.Cmd):
    prompt = "    "

    def do_exit(self, _):
        return True

    def default(self, line):
        words = form_words(line)
        try:
            words = spell_words(words)
        except SpellingError as e:
            print(e, file=sys.stderr)
            return
        try:
            words = evaluate_words(words)
            print_words(words)
        except BaseJError as e:
            print(e, file=sys.stderr)

    def do_EOF(self, _):
        return True


def main():
    try:
        Shell().cmdloop()
    except EOFError:
        return None


if __name__ == "__main__":
    main()
