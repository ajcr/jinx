import cmd

from jinx.word_formation import form_words
from jinx.word_spelling import spell_words
from jinx.word_evaluation import evaluate_words, print_words


class Shell(cmd.Cmd):
    prompt = "    "

    def do_exit(self, _):
        return True

    def default(self, line):
        words = form_words(line)
        words = spell_words(words)
        words = evaluate_words(words)
        print_words(words)


def main():
    Shell().cmdloop()


if __name__ == "__main__":
    main()
