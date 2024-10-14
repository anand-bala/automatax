import math
from abc import abstractmethod
from functools import reduce
from typing import Generic, TypeVar, final

_S = TypeVar("_S")


class AbstractAlgebra(Generic[_S]):
    """Base algebra class"""

    @staticmethod
    @abstractmethod
    def zero() -> _S:
        """Return the additive identity."""

    @staticmethod
    @abstractmethod
    def one() -> _S:
        """Return the multiplicative identity."""

    @staticmethod
    @abstractmethod
    def negate(x: _S) -> _S:
        """An involution in the algebra"""

    @staticmethod
    @abstractmethod
    def add(x: _S, y: _S) -> _S:
        """Return the addition of two elements in the algebra"""

    @staticmethod
    @abstractmethod
    def multiply(x: _S, y: _S) -> _S:
        """Return the multiplication of two elements in the algebra"""

    @classmethod
    def sum(cls, *xs: _S) -> _S:
        """Return the addition of two or more elements in the algebra"""
        return reduce(cls.add, xs, cls.zero())

    @classmethod
    def product(cls, *xs: _S) -> _S:
        """Return the multiplication of two or more elements in the algebra"""
        return reduce(cls.multiply, xs, cls.zero())


@final
class Boolean(AbstractAlgebra[bool]):

    @staticmethod
    def zero() -> bool:
        return False

    @staticmethod
    def one() -> bool:
        return True

    @staticmethod
    def negate(x: bool) -> bool:
        return not x

    @staticmethod
    def add(x: bool, y: bool) -> bool:
        return x or y

    @staticmethod
    def multiply(x: bool, y: bool) -> bool:
        return x and y


@final
class MinMax(AbstractAlgebra[float]):

    @staticmethod
    def zero() -> float:
        return -math.inf

    @staticmethod
    def one() -> float:
        return math.inf

    @staticmethod
    def negate(x: float) -> float:
        return -x

    @staticmethod
    def add(x: float, y: float) -> float:
        return max(x, y)

    @staticmethod
    def multiply(x: float, y: float) -> float:
        return min(x, y)


@final
class Lukasiewicz(AbstractAlgebra[float]):
    @staticmethod
    def zero() -> float:
        return 0.0

    @staticmethod
    def one() -> float:
        return 1.0

    @staticmethod
    def negate(x: float) -> float:
        return 1 - x

    @staticmethod
    def add(x: float, y: float) -> float:
        return min(1, x + y)

    @staticmethod
    def multiply(x: float, y: float) -> float:
        return max(0, x + y - 1.0)
