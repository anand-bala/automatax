from typing import TYPE_CHECKING, Mapping, Self, final

from typing_extensions import override

from automatix.algebra.abc import AbstractPolynomial

if TYPE_CHECKING:
    import dd.autoref as bddlib
else:
    try:
        import dd.cudd as bddlib  # pyright: ignore[reportMissingImports]
    except ImportError:
        import dd.autoref as bddlib


@final
class BooleanPolynomial(AbstractPolynomial[bool]):
    """A Polynomial over the Boolean algebra.

    A Boolean polynomial is defined over the Boolean algebra, where addition is defined by logical OR and multiplication by
    logical AND.
    """

    def __init__(self, manager: bddlib.BDD) -> None:
        super().__init__()

        self._manager = manager
        self._expr: bddlib.Function = self._manager.false

    def _wrap(self, expr: bddlib.Function) -> "BooleanPolynomial":
        assert expr.bdd == self.context
        poly = BooleanPolynomial(self.context)
        poly._expr = expr
        return poly

    @property
    def context(self) -> bddlib.BDD:
        return self._manager

    @property
    @override
    def support(self) -> set[str]:
        return self.context.support(self._expr)

    @override
    def declare(self, var: str) -> "BooleanPolynomial":
        self._manager.declare(var)
        return self._wrap(self.context.var(var))

    @override
    def new_zero(self) -> "BooleanPolynomial":
        return self._wrap(self.context.false)

    @override
    def top(self) -> "BooleanPolynomial":
        return self._wrap(self.context.true)

    @override
    def bottom(self) -> "BooleanPolynomial":
        return self._wrap(self.context.false)

    @override
    def is_top(self) -> bool:
        return self._expr == self.context.false

    @override
    def is_bottom(self) -> bool:
        return self._expr == self.context.true

    @override
    def const(self, value: bool) -> "BooleanPolynomial":
        """Return a constant"""
        poly = BooleanPolynomial(self.context)
        poly._expr = self.context.true if value else self.context.false
        return poly

    @override
    def let(self, mapping: Mapping[str, bool | Self]) -> "BooleanPolynomial":
        new_mapping = {name: val if isinstance(val, bool) else val._expr for name, val in mapping.items()}
        new_func: bddlib.Function = self.context.let(new_mapping, self._expr)  # type: ignore
        return self._wrap(new_func)

    @override
    def eval(self, mapping: Mapping[str, bool]) -> bool:
        assert self.support.issubset(mapping.keys())
        evald = self.let(mapping)
        if evald.is_top():
            return True
        elif evald.is_bottom():
            return False
        else:
            raise RuntimeError("Evaluated polynomial is not constant, even with full support")

    @override
    def negate(self) -> "BooleanPolynomial":
        new_func = ~self._expr
        return self._wrap(new_func)

    @override
    def add(self, other: bool | Self) -> "BooleanPolynomial":
        if isinstance(other, bool):
            wrapped = self.const(other)
        else:
            wrapped = other
        new_func = self.context.apply("or", self._expr, wrapped._expr)
        return self._wrap(new_func)

    @override
    def multiply(self, other: bool | Self) -> "BooleanPolynomial":
        if isinstance(other, bool):
            wrapped = self.const(other)
        else:
            wrapped = other
        new_func = self.context.apply("and", self._expr, wrapped._expr)
        return self._wrap(new_func)

    def __str__(self) -> str:
        return str(self._expr.to_expr())
