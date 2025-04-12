# Jinx

J interpreter written in Python, backed with NumPy for execution (with scope to execute with JAX, PyTorch, or another library in future).

A work in progress.

## Examples

Start the interactive shell with `jinx`.

Atoms and (multidimensional) arrays:
```j
(jinx) 3
3

(jinx) 3 5 7 11 13
[3 5 7 11 13]

(jinx) 3 2 $ 7 1 0
[[ 7  1]
 [ 0  7]
 [ 1  0]]
```

Monadic and dyadic application of verbs (applied right-to-left, parentheses force precedence):
```j
(jinx) -3.141
_3.141

(jinx) 10 - 3 5 7 11 13
[7 5 3 _1 _3]

(jinx) 8 % 4 - 2
4

(jinx) (8 % 4) - 2
0
```

## Motivation

This project is primarily a learning exercise: I want to improve my patchy understanding of J by implementing a useful subset of the language and its core concepts (ranks, obverses, etc.).

It is also an attempt to prototype an interpreter for an array language that can be executed using different array frameworks (NumPy, JAX, PyTorch, etc.) according to the user's choice.