from abc import abstractmethod

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Num, Scalar

from automatix.nfa.semirings import AbstractSemiring


class AbstractPredicate(eqx.Module, strict=True):
    """A predicate is an *effective Boolean alphabet* over some domain, e.g., real valued vectors, etc."""

    @abstractmethod
    def is_true(self, x: Num[Array, "..."]) -> bool:
        """Given a domain vector, return `True` if the predicate evaluates to true, and `False` otherwise."""
        ...

    @abstractmethod
    def weight(self, x: Num[Array, "..."], negate: bool = False) -> Scalar:
        """Scalar function that outputs the weight of an input domain vector with respect to the predicate.

        If `negate` is `True`, the predicate is logically negated. This is useful when the predicate is in NNF form.

        !!! note

            To use this function with `jax.jit`, use the `static_argnums` or `static_argnames` argument as follows:

            ```python
            from functools import partialmethod
            import jax

            p = Predicate(...)
            p_weight = jax.jit(p.weight, static_argnames=['negate'])
            ```
        """
        ...


class Not(AbstractPredicate):
    r"""Logical negation of a predicate.

    Simplified wrapper around the `weight` with `negate=True`.
    """

    arg: AbstractPredicate

    def is_true(self, x: Num[Array, "..."]) -> bool:
        return not self.arg.is_true(x)

    def weight(self, x: Num[Array, "..."], negate: bool = False) -> Scalar:
        negate = not negate  # Invert again if negate is provided
        return jax.jit(self.arg.weight)(static_argnames=["negate"])(x, negate=negate)


class And(AbstractPredicate):
    r"""Conjunction of two predicates.

    Given a semiring, the weights for two predicates \(a\) and \(b\) are computed as \(a \otimes b\).
    """

    semiring: AbstractSemiring
    args: list[AbstractPredicate]

    def is_true(self, x: Num[Array, "..."]) -> bool:
        args = [arg.is_true(x) for arg in self.args]
        return all(args)

    def weight(self, x: Num[Array, "..."], negate: bool = False) -> Scalar:
        if negate:
            raise RuntimeError("Using And/Or expressions requires negation normal form")
        weights: list[Scalar] = [jax.jit(arg.weight, static_argnames=["negate"])(x) for arg in self.args]
        return self.semiring.prod(jnp.asarray(weights))


class Or(AbstractPredicate):
    r"""Disjunction of two predicates.

    Given a semiring, the weights for two predicates \(a\) and \(b\) are computed as \(a \oplus b\).
    """

    semiring: AbstractSemiring
    args: list[AbstractPredicate]

    def is_true(self, x: Num[Array, "..."]) -> bool:
        args = [arg.is_true(x) for arg in self.args]
        return any(args)

    def weight(self, x: Num[Array, "..."], negate: bool = False) -> Scalar:
        if negate:
            raise RuntimeError("Using And/Or expressions requires negation normal form")
        weights: list[Scalar] = [jax.jit(arg.weight, static_argnames=["negate"])(x, negate) for arg in self.args]
        return self.semiring.sum(jnp.asarray(weights))
