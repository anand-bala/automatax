from dataclasses import dataclass
from typing import Callable, Generic, Iterable, Mapping, Optional, TypeVar

from automatix.algebra.abc import AbstractPolynomial, AbstractSemiring

Alph = TypeVar("Alph")
Q = TypeVar("Q")
K = TypeVar("K", bound=AbstractSemiring)
Poly = TypeVar("Poly", bound=AbstractPolynomial)


@dataclass
class Transition(Generic[Alph, Q, K, Poly]):
    src: Q
    dst: Callable[[Alph], Poly]


class AFA(Generic[Alph, Q, Poly, K]):

    def __init__(self) -> None:
        self._transitions: dict[Q, Transition] = dict()
        self._initial: Optional[Poly] = None
        self._final: dict[Q, K] = dict()

    def add_location(self, location: Q, transition: Transition) -> None:
        """Add a location in the automaton and the transition function at the location."""
        if location in self._transitions:
            raise ValueError(f"Location {location} already exists in automaton")
        self._transitions[location] = transition

    @property
    def transitions(self) -> Mapping[Q, Transition]:
        """Get the transition relation"""
        return self._transitions

    @property
    def locations(self) -> Iterable[Q]:
        return self.transitions.keys()

    @property
    def initial_location(self) -> Optional[Poly]:
        return self._initial

    @initial_location.setter
    def set_initial(self, q0: Poly) -> None:
        self._initial = q0

    @property
    def final_mapping(self) -> Mapping[Q, K]:
        return self._final

    @final_mapping.setter
    def set_final(self, q_f: Mapping[Q, K]) -> None:
        self._final = dict(q_f.items())

    def __getitem__(self, key: Q) -> Transition:
        return self.transitions[key]
