# Jinx

A work-in-progress interpreter for the J programming language.

Currently supports many features that are central to J, including:
- Multidimensional arrays of integers and floats.
- Many primitive verbs (e.g. `+`, `%:`, `,`, ...), adverbs (`/`, `~`, ...) and conjunctions (`"`, `@:`, ...).
- Correct monadic and dyadic application of verbs of different ranks.
- Obverses.
- Trains (hooks and forks).

This allows some fairly sophisticated tacit expressions to be evaluated.

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

## Motivation / Warning

This project is primarily a learning exercise. I want to improve my patchy understanding of J by straightforwardly implementing a useful subset of it. Understand how J code is evaluated. Nail down how rank works. And so on.

The code is my mental map of information grabbed from pages of language documentation, book chapters, and forum posts into something that resembles J. There are references to these sources scattered throughout the code.

Recreating the sophistication and decades of attention to detail that have gone into making the official J interpreter so feature rich and performant is not the goal here. There will be bugs, missing parts and glaring performance gaps.

But the advantage of this learn-by-making approach is that the wonderful concepts of the J language (and other APL dialects) have the potential to be mixed and matched with other tools and frameworks I know more about.

For example, in this code the separation between "interpreting J code" and "actual execution on array-like objects" is very clearly separated. At the moment "array-like objects" and "execution" are NumPy arrays and methods, but could very easily be swapped out for PyTorch, JAX, or any other array framework. This sets up experiments in JIT compilation, execution on accelerators and other fun adventures.