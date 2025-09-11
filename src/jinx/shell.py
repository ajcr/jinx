import cmd
import sys

from jinx.errors import BaseJError, SpellingError
from jinx.word_formation import form_words
from jinx.word_spelling import spell_words
from jinx.word_evaluation import evaluate_words, print_words


class Shell(cmd.Cmd):
    prompt = "    "

    def __init__(self):
        super().__init__()
        self.variables = {}

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
            words = evaluate_words(words, self.variables)
            print_words(words, self.variables)
        except BaseJError as error:
            print(f"{type(error).__name__}: {error}", file=sys.stderr)

    def do_EOF(self, _):
        return True

    # '?' is a primitive verb in J and we want the Cmd class to disregard it.
    # and not treat it as a help command.
    def do_help(self, line):
        return self.default("?" + line)


def main():
    try:
        Shell().cmdloop()
    except EOFError:
        return None


if __name__ == "__main__":
    main()
