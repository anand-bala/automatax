from abc import ABC, abstractmethod
from typing import Generic, Mapping, Self, TypeVar

from typing_extensions import ClassVar

S = TypeVar("S")


class AbstractSemiring(ABC, Generic[S]):
    """Base semiring class"""

    @staticmethod
    @abstractmethod
    def zero() -> S:
        """Return the additive identity"""

    @staticmethod
    @abstractmethod
    def one() -> S:
        """Return the multiplicative identity"""

    @staticmethod
    @abstractmethod
    def add(x: S, y: S) -> S:
        """Return the addition of two elements in the semiring"""

    @staticmethod
    @abstractmethod
    def multiply(x: S, y: S) -> S:
        """Return the multiplication of two elements in the semiring"""

    is_additively_idempotent: ClassVar[bool] = False
    is_multiplicatively_idempotent: ClassVar[bool] = False
    is_commutative: ClassVar[bool] = False
    is_simple: ClassVar[bool] = False


class AbstractNegation(ABC, Generic[S]):
    """A negation function on `S` is an involution on `S`"""

    @staticmethod
    @abstractmethod
    def negate(x: S) -> S:
        """An involution in the algebra"""


class AbstractPolynomial(ABC, Generic[S]):
    """A polynomial with coefficients and the value of variables in `S`, where `S` is a semiring."""

    @property
    @abstractmethod
    def support(self) -> set[str]:
        """Return the list of variables with non-zero coefficients in the polynomial"""
        ...

    @abstractmethod
    def declare(self, var: str) -> Self:
        """Declare a variable for the polynomial."""

    @abstractmethod
    def new_zero(self) -> Self:
        """Return a new constant `0` polynomial"""

    @abstractmethod
    def top(self) -> Self:
        """Return the multiplicative identity of the polynomial ring"""

    @abstractmethod
    def bottom(self) -> Self:
        """Return the additive identity of the polynomial ring"""

    @abstractmethod
    def is_bottom(self) -> bool:
        """Returns `True` if the Polynomial is just the additive identity in the ring."""

    @abstractmethod
    def is_top(self) -> bool:
        """Returns `True` if the Polynomial is just the multiplicative identity in the ring."""

    @abstractmethod
    def const(self, value: S) -> Self:
        """Return a new constant polynomial with value"""

    @abstractmethod
    def let(self, mapping: Mapping[str, S | Self]) -> Self:
        """Substitute variables with constants or other polynomials."""

    @abstractmethod
    def eval(self, mapping: Mapping[str, S]) -> S:
        """Evaluate the polynomial with the given variable values.

        !!! note

            Asserts that all variables that form the support of the polynomial are used.
        """

    @abstractmethod
    def negate(self) -> Self:
        """return the negation of the polynomial"""

    @abstractmethod
    def add(self, other: S | Self) -> Self:
        """Return the addition (with appropriate ring) of two polynomials."""

    @abstractmethod
    def multiply(self, other: S | Self) -> Self:
        """Return the multiplication (with appropriate ring) of two polynomials."""

    def __add__(self, other: S | Self) -> Self:
        return self.add(other)

    def __radd__(self, other: S | Self) -> Self:
        return self.add(other)

    def __mul__(self, other: S | Self) -> Self:
        return self.multiply(other)

    def __rmul__(self, other: S | Self) -> Self:
        return self.multiply(other)

    def __call__(self, mapping: Mapping[str, S | Self]) -> S | Self:
        return self.let(mapping)
