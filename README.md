# Jinx

A work-in-progress interpreter for the J programming language.

Written in Python and NumPy, with the potential to execute via other Python array/tensor frameworks in future.

Currently supports many features that are central to J, including:
- Multidimensional arrays of integers and floats.
- Many primitive verbs (e.g. `+`, `%:`, `,`, ...), adverbs (`/`, `~`, ...) and conjunctions (`"`, `@:`, ...).
- Correct monadic and dyadic application of verbs of different ranks.
- Obverses.
- Trains (hooks and forks).

This allows some fairly sophisticated tacit expressions to be evaluated...

## Examples

Start the interactive shell with `jinx`. As in the official J implementation, the shell prompt is four spaces, so commands appear indented.

- The "trapping rainwater" problem (solution from [here](https://mmapped.blog/posts/04-square-joy-trapped-rain-water)):
```j
    +/@((>./\ <. >./\.)-]) 0 1 0 2 1 0 1 3 2 1 2 1
6
```
- The correlation between two arrays of numbers (taken from [here](https://stackoverflow.com/a/44845495/3923281)):
```j
    2 1 1 7 9 (+/@:* % *&(+/)&.:*:)&(- +/%#) 6 3 1 5 7
0.721332
```

## Motivation

This project is primarily a learning exercise: I want to improve my patchy understanding of J by implementing a useful subset of the language and its core concepts.

It is also an attempt to prototype an interpreter for an array language that can be executed using Python's different array and tensor frameworks (NumPy, JAX, PyTorch, etc.) according to user's wishes. It is immensely satisfying to build complex expressions using a few symbols and, with no further effort, execute the expression over massive arrays of numbers on an accelerator.