from abc import abstractmethod

import equinox as eqx
from jaxtyping import Array, Num


class Predicate(eqx.Module):
    """A predicate is an *effective Boolean alphabet* over some domain, e.g., real valued vectors, etc."""

    @abstractmethod
    def is_true(self, x: Num[Array, "..."]) -> bool:
        """Given a domain vector, return `True` if the predicate evaluates to true, and `False` otherwise."""
        ...

    @abstractmethod
    def weight(self, x: Num[Array, "..."]) -> Num[Array, ""]:
        """Scalar function that outputs the weight of an input domain vector with respect to the predicate."""
        ...
