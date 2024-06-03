from typing import Callable

import equinox as eqx
import jax
import jax.numpy as jnp
import networkx as nx
from jaxtyping import Array, Num

from automatax.predicate import Predicate
from automatax.semirings import AbstractSemiring


class Automaton:

    def __init__(self) -> None:
        self._graph = nx.DiGraph()
        self._initial_location: set[int] = set()
        self._final_locations: set[int] = set()

    def add_location(self, location: int, initial: bool = False, final: bool = False) -> None:
        if location in self._graph.nodes:
            raise ValueError(f"Location {location} already exists in automaton")
        if initial:
            self._initial_location.add(location)
        if final:
            self._final_locations.add(location)
        self._graph.add_node(location, initial=initial, final=final)

    def add_transition(self, src: int, dst: int, guard: Predicate) -> None:
        if (src, dst) in self._graph.edges:
            raise ValueError(f"Transition from {src} to {dst} already exists. Did you want to update the guard?")
        self._graph.add_edge(src, dst, guard=guard)

    @property
    def num_locations(self) -> int:
        return len(self._graph)

    def __len__(self) -> int:
        return self.num_locations

    @property
    def initial_locations(self) -> set[int]:
        return self._initial_location

    @property
    def final_locations(self) -> set[int]:
        return self._final_locations


class AutomatonOperator(eqx.Module):
    initial_weights: Num[Array, " q"]
    final_weights: Num[Array, " q"]
    cost_transitions: Callable[[Num[Array, "..."]], Num[Array, "q q"]]


def make_automaton_operator(aut: Automaton, semiring: AbstractSemiring) -> AutomatonOperator:

    initial_weights = semiring.zeros(aut.num_locations).at[jnp.array(list(aut.initial_locations))].set(semiring.ones(1).item())
    final_weights = semiring.zeros(aut.num_locations).at[jnp.array(list(aut.final_locations))].set(semiring.ones(1).item())
    n_q = aut.num_locations

    @jax.jit
    def cost_transitions(x: Num[Array, "..."]) -> Num[Array, " {n_q} {n_q}"]:
        src: int
        dst: int
        guard: Predicate
        matrix = semiring.zeros((n_q, n_q))
        for src, dst, guard in aut._graph.edges.data("guard"):
            matrix.at[src, dst].set(guard.weight(x))

        return matrix

    return AutomatonOperator(initial_weights=initial_weights, final_weights=final_weights, cost_transitions=cost_transitions)
