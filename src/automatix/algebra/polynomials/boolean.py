from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Iterable, Mapping, Optional, Self, final

from typing_extensions import override

from automatix.algebra.abc import AbstractPolynomial

if TYPE_CHECKING:
    import dd.autoref as bddlib
else:
    try:
        import dd.cudd as bddlib  # pyright: ignore[reportMissingImports]
    except ImportError:
        import dd.autoref as bddlib


@dataclass
class Context:
    variables: set[str] = field(default_factory=set)
    bdd: bddlib.BDD = field(default_factory=lambda: bddlib.BDD())

    def __post_init__(self) -> None:
        self.declare(self.variables)

    def declare(self, variables: Iterable[str]) -> None:
        """Add variables to the polynomial context."""
        new_vars = set(variables) - self.variables
        if len(new_vars) > 0:
            self.bdd.declare(*new_vars)
            self.variables.update(new_vars)


@final
class BooleanPolynomial(AbstractPolynomial[bool]):
    """A Polynomial over the Boolean algebra.

    A Boolean polynomial is defined over the Boolean algebra, where addition is defined by logical OR and multiplication by
    logical AND.
    """

    def __init__(
        self,
        manager: Optional[Context] = None,
        *,
        variables: Optional[Iterable[str]] = None,
    ) -> None:
        super().__init__()

        variables = variables or {}
        if manager:
            self._manager = manager
            self._manager.declare(set(variables))
        else:
            self._manager = Context(variables=set(variables))
        self._expr: bddlib.Function = self._bdd.true

    @property
    def _bdd(self) -> bddlib.BDD:
        return self._manager.bdd

    @classmethod
    def _wrap(cls, expr: bddlib.Function, manager: Context) -> "BooleanPolynomial":
        poly = BooleanPolynomial(manager=manager)
        poly._expr = expr
        return poly

    @property
    def context(self) -> Context:
        return self._manager

    @property
    @override
    def support(self) -> set[str]:
        return self._bdd.support(self._expr)

    @override
    def declare(self, vars: Iterable[str]) -> Iterable["BooleanPolynomial"]:
        self._manager.declare(vars)
        poly_vars = [self._wrap(self._bdd.var(v), self._manager) for v in vars]
        return poly_vars

    @classmethod
    def zero(cls) -> "BooleanPolynomial":
        manager = Context()
        return cls._wrap(manager.bdd.false, manager)

    @override
    @classmethod
    def const(cls, value: bool, manager: Optional[Context] = None) -> "BooleanPolynomial":
        """Return a constant"""
        poly = BooleanPolynomial(manager=manager)
        if value is True:
            poly._expr = poly._bdd.true
        else:
            poly._expr = poly._bdd.false
        return poly

    @override
    def let(self, mapping: Mapping[str, bool | Self]) -> "BooleanPolynomial":
        new_mapping = {name: val if isinstance(val, bool) else val._expr for name, val in mapping.items()}
        new_func: bddlib.Function = self._bdd.let(new_mapping, self._expr)  # type: ignore
        return self._wrap(new_func, self._manager)

    @override
    def eval(self, mapping: Mapping[str, bool]) -> bool:
        assert self.support.issubset(mapping.keys())
        evald = self.let(mapping)
        if evald == self._bdd.true:
            return True
        elif evald == self._bdd.false:
            return False
        else:
            raise RuntimeError("Evaluated polynomial is not constant, even with full support")

    @override
    def add(self, other: bool | Self) -> "BooleanPolynomial":
        if isinstance(other, bool):
            wrapped = self.const(other, manager=self._manager)
        else:
            wrapped = other
        new_func = self._expr | wrapped._expr
        return self._wrap(new_func, self._manager)

    @override
    def multiply(self, other: bool | Self) -> "BooleanPolynomial":
        if isinstance(other, bool):
            wrapped = self.const(other, manager=self._manager)
        else:
            wrapped = other
        new_func = self._expr & wrapped._expr
        return self._wrap(new_func, self._manager)

    @override
    def is_top(self) -> bool:
        return self._expr == self._bdd.false

    @override
    def is_bottom(self) -> bool:
        return self._expr == self._bdd.true
