"""Transform STREL parse tree to an AFA."""

from dataclasses import dataclass
from functools import partial
from typing import Callable, Generic, TypeAlias, TypeVar

import networkx as nx

import automatix.logic.strel as strel
from automatix.afa.automaton import AFA, AbstractTransition
from automatix.algebra.abc import AbstractPolynomial

K = TypeVar("K")

Location: TypeAlias = int

Alph: TypeAlias = "nx.Graph[Location]"
"""Input alphabet is a graph over location vertices, with distance edge weights and vertex labels corresponding to semiring
values for each predicate"""

Q: TypeAlias = tuple[strel.Expr, Location]
"""Each state in the automaton represents a subformula in the specification and an ego location.
"""

Poly: TypeAlias = AbstractPolynomial[K]
LabellingFn: TypeAlias = Callable[[Alph, Location, str], K]


@dataclass
class Transitions(AbstractTransition[Alph, Q, K]):
    mapping: dict[Q, Callable[[Alph], Poly[K]]]

    def __call__(self, input: Alph, state: Q) -> Poly[K]:
        fn = self.mapping[state]
        return fn(input)


class StrelAutomaton(AFA[Alph, Q, K]):
    """(Weighted) Automaton for STREL"""

    def __init__(
        self,
        transitions: Transitions[K],
        expr_var_map: dict[Q, Poly[K]],
        var_node_map: dict[str, Q],
    ) -> None:
        # assert set(transitions.mapping.keys()) == set(expr_var_map.keys())
        super().__init__(transitions)
        self.expr_var_map = expr_var_map
        self.var_node_map = var_node_map

    def next(self, input: Alph, current: Poly[K]) -> Poly[K]:
        """Get the polynomial after transitions by evaluating the current polynomial with the transition function."""

        transitions = {var: self.transitions(input, self.var_node_map[var]) for var in current.support}
        new_state = current.let(transitions)
        return new_state

    @classmethod
    def from_strel_expr(
        cls,
        phi: strel.Expr,
        label_fn: LabellingFn[K],
        polynomial: Poly[K],
        max_locs: int,
    ) -> "StrelAutomaton":
        """Convert a STREL expression to an AFA with the given alphabet"""

        visitor = _ExprMapper(label_fn, polynomial, max_locs)
        visitor.visit(phi)

        transitions = Transitions(visitor.transitions)
        aut = cls(transitions, visitor.expr_var_map, visitor.var_node_map)

        return aut


class _ExprMapper(Generic[K]):
    """Post-order visitor for creating transitions"""

    def __init__(self, label_fn: LabellingFn[K], polynomial: Poly[K], max_locs: int) -> None:
        assert max_locs > 0, "STREL graphs should have at least 1 location"
        self.max_locs = max_locs
        self.label_fn = label_fn
        # Create a const polynomial for tracking nodes
        self.manager = polynomial
        # Maps the string representation of a subformula to the AFA node
        # This is also the visited states.
        self.expr_var_map: dict[Q, Poly[K]] = dict()
        # Maps the transition relation
        self.transitions: dict[Q, Callable[[Alph], Poly[K]]] = dict()
        # Map from the polynomial var string to the state in Q
        self.var_node_map: dict[str, Q] = dict()

    def _add_aut_state(self, phi: strel.Expr) -> None:
        phi_str = str(phi)
        not_phi = ~phi
        not_phi_str = str(not_phi)
        for loc in range(self.max_locs):
            self.expr_var_map.setdefault(
                (phi, loc),
                self.manager.declare(str((phi_str, loc))),
            )
            self.var_node_map.setdefault(str((phi_str, loc)), (phi, loc))
            self.expr_var_map.setdefault(
                (not_phi, loc),
                self.manager.declare(str((not_phi_str, loc))),
            )
            self.var_node_map.setdefault(str((not_phi_str, loc)), (not_phi, loc))

    def _add_expr_alias(self, phi: strel.Expr, alias: strel.Expr) -> None:
        phi_str = str(phi)
        not_phi = ~phi
        not_phi_str = str(not_phi)
        for loc in range(self.max_locs):
            self.expr_var_map[(phi, loc)] = self.expr_var_map[(alias, loc)]
            self.transitions[(phi, loc)] = self.transitions[(alias, loc)]
            self.var_node_map.setdefault(str((phi_str, loc)), (phi, loc))
            self.expr_var_map[(~phi, loc)] = self.expr_var_map[(~alias, loc)]
            self.transitions[(phi, loc)] = self.transitions[(alias, loc)]
            self.var_node_map.setdefault(str((not_phi_str, loc)), (not_phi, loc))

    def _add_temporal_transition(self, phi: strel.Expr, transition: Callable[[Location, Alph], Poly[K]]) -> None:
        for loc in range(self.max_locs):
            self.transitions.setdefault(
                (phi, loc),
                partial(transition, loc),
            )

    def _expand_add_next(self, phi: strel.NextOp) -> None:
        if phi.steps is None:
            steps = 1
        else:
            steps = phi.steps
        # Expand the formula into nested nexts
        for i in range(steps, 0, -1):
            expr = strel.NextOp(i, phi.arg)
            self._add_aut_state(expr)

        # Add the temporal transition while there is a nested next
        for i in range(steps, 1, -1):  # doesn't include 1
            expr = strel.NextOp(i, phi.arg)
            sub_expr = strel.NextOp(i - 1, phi.arg)
            self._add_temporal_transition(
                expr,
                lambda loc, _, sub_expr=sub_expr: self.expr_var_map[(sub_expr, loc)],
            )
        # Add the final bit where there is no nested next
        self._add_temporal_transition(phi, lambda loc, _, arg=phi.arg: self.expr_var_map[(arg, loc)])

    def _expand_add_globally(self, phi: strel.GloballyOp) -> None:
        # G[a,b] phi = ~F[a,b] ~phi
        expr = ~strel.EventuallyOp(phi.interval, ~phi.arg)
        self.visit(expr)
        self._add_expr_alias(phi, expr)

    def _expand_add_eventually(self, phi: strel.EventuallyOp) -> None:
        # F[a,b] phi = X X ... X (phi | X (phi | X( ... | X f)))
        #              ^^^^^^^^^        ^^^^^^^^^^^^^^^^^^^^^^^
        #               a times                 b-a times
        #            = X[a] (phi | X (phi | X( ... | X f)))
        #                          ^^^^^^^^^^^^^^^^^^^^^^^
        #                                  b-a times
        match phi.interval:
            case None | strel.TimeInterval(None, None) | strel.TimeInterval(0, None):
                # phi = F arg
                # Return as is
                self._add_aut_state(phi)
                # Expand as F arg = arg | X F arg
                self._add_temporal_transition(
                    phi,
                    lambda loc, alph: self.transitions[(phi.arg, loc)](alph) + self.expr_var_map[(phi, loc)],
                )
            case strel.TimeInterval(0 | None, int(t2)):
                # phi = F[0, t2] arg
                for i in range(t2, 0, -1):
                    expr = strel.EventuallyOp(strel.TimeInterval(0, i), phi.arg)
                    self._add_aut_state(expr)
                # Expand as F[0, t2] arg = arg | X F[0, t2-1] arg
                for i in range(t2, 1, -1):  # Ignore F[0, 1] arg
                    expr = strel.EventuallyOp(strel.TimeInterval(0, i), phi.arg)
                    sub_expr = strel.EventuallyOp(strel.TimeInterval(0, i - 1), phi.arg)
                    self._add_temporal_transition(
                        expr,
                        lambda loc, alph, sub_expr=sub_expr: self.transitions[(phi.arg, loc)](alph)
                        + self.expr_var_map[(sub_expr, loc)],
                    )
            case strel.TimeInterval(int(t1), None):
                # phi = F[t1,] arg = X[t1] F arg
                expr = strel.NextOp(t1, strel.EventuallyOp(None, phi.arg))
                self.visit(expr)
                self._add_expr_alias(phi, expr)

            case strel.TimeInterval(int(t1), int(t2)):
                # phi = F[t1, t2] arg = X[t1] F[0, t2 - t1] arg
                expr = strel.NextOp(
                    t1,
                    strel.EventuallyOp(
                        strel.TimeInterval(0, t2 - t1),
                        phi.arg,
                    ),
                )
                self.visit(expr)
                self._add_expr_alias(phi, expr)

    def _expand_add_until(self, phi: strel.UntilOp) -> None:
        # lhs U[t1,t2] rhs = (F[t1,t2] rhs) & (lhs U[t1,] rhs)
        # lhs U[t1,  ] rhs = ~F[0,t1] ~(lhs U rhs)
        match phi.interval:
            case None | strel.TimeInterval(0, None) | strel.TimeInterval(None, None):
                # phi = lhs U rhs
                self._add_aut_state(phi)
                # Expand as phi = lhs U rhs = rhs | (lhs & X phi)
                self._add_temporal_transition(
                    phi,
                    lambda loc, alph: self.transitions[(phi.rhs, loc)](alph)
                    + (self.transitions[(phi.lhs, loc)](alph) * self.expr_var_map[(phi, loc)]),
                )
            case strel.TimeInterval(int(t1), None):
                # phi = lhs U[t1,] rhs = ~F[0,t1] ~(lhs U rhs)
                expr = ~strel.EventuallyOp(
                    strel.TimeInterval(0, t1),
                    ~strel.UntilOp(phi.lhs, None, phi.rhs),
                )
                self.visit(expr)
                self._add_expr_alias(phi, expr)
            case strel.TimeInterval(int(t1), int()):
                # phi = lhs U[t1,t2] rhs = (F[t1,t2] rhs) & (lhs U[t1,] rhs)
                expr = strel.AndOp(
                    strel.EventuallyOp(phi.interval, phi.rhs),
                    strel.UntilOp(
                        interval=strel.TimeInterval(t1, None),
                        lhs=phi.lhs,
                        rhs=phi.rhs,
                    ),
                )
                self.visit(expr)
                self._add_expr_alias(phi, expr)

    def visit(self, phi: strel.Expr) -> None:
        # Skip if phi already visited
        if str(phi) in self.expr_var_map.keys():
            return
        # 1. If phi is not a leaf expression visit its Expr children
        # 2. Add phi and ~phi as AFA nodes
        # 3. Add the transition for phi and ~phi
        match phi:
            case strel.Identifier():
                self._add_aut_state(phi)
                # The transition is to evaluate it.
                self._add_temporal_transition(
                    phi,
                    lambda loc, alph: self.manager.const(self.label_fn(alph, loc, phi.name)),
                )
            case strel.NotOp(arg):
                # Just add the argument as the negation will be added implicitely
                self.visit(arg)
                self._add_temporal_transition(
                    phi,
                    lambda loc, alph: self.transitions[(arg, loc)](alph).negate(),
                )
            case strel.AndOp(lhs, rhs):
                self.visit(lhs)
                self.visit(rhs)
                self._add_aut_state(phi)
                self._add_temporal_transition(
                    phi,
                    lambda loc, alph: self.transitions[(lhs, loc)](alph) * self.transitions[(rhs, loc)](alph),
                )
            case strel.OrOp(lhs, rhs):
                self.visit(lhs)
                self.visit(rhs)
                self._add_aut_state(phi)
                self._add_temporal_transition(
                    phi,
                    lambda loc, alph: self.transitions[(lhs, loc)](alph) + self.transitions[(rhs, loc)](alph),
                )
            case strel.EverywhereOp(interval, arg):
                self.visit(arg)
                self._add_aut_state(phi)
                # TODO: do something for the everywhere closure
                raise NotImplementedError()
            case strel.SomewhereOp(interval, arg):
                self.visit(arg)
                self._add_aut_state(phi)
                # TODO: do something for the everywhere closure
                raise NotImplementedError()
            case strel.EscapeOp(interval, arg):
                self.visit(arg)
                self._add_aut_state(phi)
                raise NotImplementedError()
            case strel.ReachOp(lhs, interval, rhs):
                self.visit(lhs)
                self.visit(rhs)
                self._add_aut_state(phi)
                raise NotImplementedError()
            case strel.NextOp():
                self.visit(phi.arg)
                self._expand_add_next(phi)
            case strel.GloballyOp():
                self.visit(phi.arg)
                self._expand_add_globally(phi)
            case strel.EventuallyOp():
                self.visit(phi.arg)
                self._expand_add_eventually(phi)
            case strel.UntilOp():
                self.visit(phi.lhs)
                self.visit(phi.rhs)
                self._expand_add_until(phi)
