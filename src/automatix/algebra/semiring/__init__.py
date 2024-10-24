import math

from typing_extensions import ClassVar, override

from automatix.algebra.abc import AbstractNegation, AbstractSemiring


class ElementaryAlgebra(AbstractSemiring[float], AbstractNegation[float]):
    is_commutative: ClassVar[bool] = True

    @override
    @staticmethod
    def zero() -> float:
        return 0

    @override
    @staticmethod
    def one() -> float:
        return 1

    @override
    @staticmethod
    def add(x: float, y: float) -> float:
        return x + y

    @override
    @staticmethod
    def multiply(x: float, y: float) -> float:
        return x * y

    @override
    @staticmethod
    def negate(x: float) -> float:
        return -x


class BooleanAlgebra(AbstractSemiring[bool], AbstractNegation[bool]):
    is_additively_idempotent: ClassVar[bool] = True
    is_multiplicatively_idempotent: ClassVar[bool] = True
    is_commutative: ClassVar[bool] = True
    is_simple: ClassVar[bool] = True

    @override
    @staticmethod
    def zero() -> bool:
        return False

    @override
    @staticmethod
    def one() -> bool:
        return True

    @override
    @staticmethod
    def add(x: bool, y: bool) -> bool:
        return x or y

    @override
    @staticmethod
    def multiply(x: bool, y: bool) -> bool:
        return x and y

    @override
    @staticmethod
    def negate(x: bool) -> bool:
        return not x


class MaxMinAlgebra(AbstractSemiring[float], AbstractNegation[float]):
    is_additively_idempotent: ClassVar[bool] = True
    is_multiplicatively_idempotent: ClassVar[bool] = True
    is_commutative: ClassVar[bool] = True
    is_simple: ClassVar[bool] = True

    @override
    @staticmethod
    def zero() -> float:
        return -math.inf

    @override
    @staticmethod
    def one() -> float:
        return math.inf

    @override
    @staticmethod
    def add(x: float, y: float) -> float:
        return max(x, y)

    @override
    @staticmethod
    def multiply(x: float, y: float) -> float:
        return min(x, y)

    @override
    @staticmethod
    def negate(x: float) -> float:
        return -x


class LukasiewiczAlgebra(AbstractSemiring[float], AbstractNegation[float]):
    @override
    @staticmethod
    def zero() -> float:
        return 0.0

    @override
    @staticmethod
    def one() -> float:
        return 1.0

    @override
    @staticmethod
    def negate(x: float) -> float:
        return 1 - x

    @override
    @staticmethod
    def add(x: float, y: float) -> float:
        return min(1, x + y)

    @override
    @staticmethod
    def multiply(x: float, y: float) -> float:
        return max(0, x + y - 1.0)


class MaxPlusSemiring(AbstractSemiring[float]):
    is_additively_idempotent: ClassVar[bool] = True
    is_commutative: ClassVar[bool] = True
    is_simple: ClassVar[bool] = True

    @override
    @staticmethod
    def zero() -> float:
        return -math.inf

    @override
    @staticmethod
    def one() -> float:
        return 0

    @override
    @staticmethod
    def add(x: float, y: float) -> float:
        return max(x, y)

    @override
    @staticmethod
    def multiply(x: float, y: float) -> float:
        return x + y


class MinPlusSemiring(AbstractSemiring[float]):
    is_additively_idempotent: ClassVar[bool] = True
    is_commutative: ClassVar[bool] = True
    is_simple: ClassVar[bool] = True

    @override
    @staticmethod
    def zero() -> float:
        return math.inf

    @override
    @staticmethod
    def one() -> float:
        return 0

    @override
    @staticmethod
    def add(x: float, y: float) -> float:
        return min(x, y)

    @override
    @staticmethod
    def multiply(x: float, y: float) -> float:
        return x + y
