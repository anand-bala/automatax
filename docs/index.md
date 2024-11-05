# Symbolic (Weighted) Automata Monitoring

This project implements different automata I use in my research, including
nondeterministic weighted automata and alternating weighted automata.

## Differentiable Automata in [JAX](https://github.com/google/jax)

The `automatix.nfa` module implements differentiable automata in JAX, along with
`automatix.algebra.semiring.jax_backend`.
Specifically, it does so by defining _matrix operators_ on the automata transitions,
which can then be interpreted over a semiring to yield various acceptance and weighted
semantics.

## Alternating Automata as Ring Polynomials

The `automatix.afa` module implements weighted alternating finite automata over
algebra defined in `automatix.algebra.semiring`.
