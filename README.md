# Jinx

J interpreter written in Python, backed with NumPy for execution (with scope to execute with JAX, PyTorch, or another library in future).

A work in progress.

## Examples

Start the interactive shell with `jinx`.

Atoms (scalars) and multidimensional arrays:
```j
    _3
_3

    3 5 7 11 13
[3 5 7 11 13]

    3 2 $ 7 1 0
[[ 7  1]
 [ 0  7]
 [ 1  0]]
```

Monadic and dyadic application of verbs (applied right-to-left, parentheses force precedence):
```j
     -3.141
_3.141

    10 - 3 5 7 11 13
[7 5 3 _1 _3]

    8 % 4 - 2
4

    (8 % 4) - 2
0
```

Adverbs to modify verbs (e.g. apply sum over axis of a multidimensional array):
```j
    +/ i. 2 3 4
[[ 12  14  16  18]
 [ 20  22  24  26]
 [ 28  30  32  34]]
```

## Motivation

This project is primarily a learning exercise: I want to improve my patchy understanding of J by implementing a useful subset of the language and its core concepts (ranks, obverses, etc.).

It is also an attempt to prototype an interpreter for an array language that can be executed using different array frameworks (NumPy, JAX, PyTorch, etc.) according to the user's choice.