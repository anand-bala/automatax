from abc import ABC, abstractmethod
from typing import Generic, TypeVar

from typing_extensions import ClassVar

_S = TypeVar("_S")


class AbstractSemiring(ABC, Generic[_S]):
    """Base semiring class"""

    @staticmethod
    @abstractmethod
    def zero() -> _S:
        """Return the additive identity"""

    @staticmethod
    @abstractmethod
    def one() -> _S:
        """Return the multiplicative identity"""

    @staticmethod
    @abstractmethod
    def add(x: _S, y: _S) -> _S:
        """Return the addition of two elements in the semiring"""

    @staticmethod
    @abstractmethod
    def multiply(x: _S, y: _S) -> _S:
        """Return the multiplication of two elements in the semiring"""

    is_additively_idempotent: ClassVar[bool] = False
    is_multiplicatively_idempotent: ClassVar[bool] = False
    is_commutative: ClassVar[bool] = False
    is_simple: ClassVar[bool] = False


class AbstractNegation(ABC, Generic[_S]):
    """A negation function on `_S` is an involution on `_S`"""

    @staticmethod
    @abstractmethod
    def negate(x: _S) -> _S:
        """An involution in the algebra"""
