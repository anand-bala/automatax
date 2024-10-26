from abc import ABC, abstractclassmethod, abstractmethod
from typing import Generic, Mapping, Optional, Self, TypeVar

from typing_extensions import ClassVar, override

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


Ctx = TypeVar("Ctx")


class AbstractPolynomial(ABC, Generic[S, Ctx]):
    """A polynomial with coefficients and the value of variables in `S`, where `S` is a semiring."""

    @property
    @abstractmethod
    def context(self) -> Ctx:
        """Give the polynomial's context manager"""

    @property
    @abstractmethod
    def support(self) -> set[str]:
        """Return the list of variables with non-zero coefficients in the polynomial"""
        ...

    @abstractmethod
    def declare(self, vars: set[str]) -> None:
        """Declare a list of variables that define the support of the polynomial."""

    @abstractmethod
    @classmethod
    def zero(cls, manager: Optional[Ctx] = None) -> Self:
        """Return a constant `0` polynomial"""

    @abstractmethod
    @classmethod
    def const(cls, value: bool, manager: Optional[Ctx] = None) -> Self:
        """Return a constant"""

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
    def add(self, other: S | Self) -> Self:
        """Return the addition (with appropriate semiring) of two polynomials."""

    @abstractmethod
    def multiply(self, other: S | Self) -> Self:
        """Return the multiplication (with appropriate semiring) of two polynomials."""

    @abstractmethod
    def is_top(self) -> bool:
        """Returns `True` if the Polynomial is just the additive identity in the semiring."""

    @abstractmethod
    def is_bottom(self) -> bool:
        """Returns `True` if the Polynomial is just the multiplicative identity in the semiring."""

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
