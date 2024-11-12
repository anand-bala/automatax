"""Transform STREL parse tree to an AFA."""

import math
from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import Callable, Collection, Generic, Iterable, Iterator, Mapping, Optional, TypeAlias, TypeVar

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
Manager: TypeAlias = Poly[K]
LabellingFn: TypeAlias = Callable[[Alph, Location, str], K]


@dataclass
class Transitions(AbstractTransition[Alph, Q, K]):
    mapping: dict[Q, Callable[[Alph], Poly[K]]]
    manager: Manager[K]
    label_fn: LabellingFn[K]

    def __call__(self, input: Alph, state: Q) -> Poly[K]:
        match state[0]:
            case strel.Constant(value):
                if value:
                    return self.manager.top()
                else:
                    return self.manager.bottom()
            case strel.Identifier(name):
                return self.manager.const(self.label_fn(input, state[1], name))
        fn = self.mapping[state]
        return fn(input)


class StrelAutomaton(AFA[Alph, Q, K]):
    """(Weighted) Automaton for STREL"""

    def __init__(
        self,
        initial_expr: strel.Expr,
        transitions: Transitions[K],
        expr_var_map: dict[Q, Poly[K]],
        var_node_map: dict[str, Q],
    ) -> None:
        # assert set(transitions.mapping.keys()) == set(expr_var_map.keys())
        super().__init__(transitions)
        assert expr_var_map.keys() == transitions.mapping.keys()

        self.initial_expr = initial_expr
        self.expr_var_map = expr_var_map
        self.var_node_map = var_node_map
        self._manager = next(iter(self.expr_var_map.values()))

        def _is_accepting(expr: strel.Expr) -> bool:
            return (
                isinstance(expr, strel.NotOp)
                and isinstance(expr.arg, (strel.UntilOp, strel.EventuallyOp))
                and (expr.arg.interval is None or expr.arg.interval.is_untimed())
            ) or expr == self.initial_expr

        self.accepting_states = {(expr, loc) for (expr, loc) in transitions.mapping.keys() if _is_accepting(expr)}

    def initial_at(self, loc: Location) -> Poly[K]:
        """Return the polynomial representation of the initial state"""
        return self.expr_var_map[(self.initial_expr, loc)]

    @property
    def final_mapping(self) -> Mapping[str, K]:
        """Return the weights/labels for the final/accepting states"""
        return {
            str((str(phi), loc)): (
                self._manager.bottom().eval({}) if (phi, loc) not in self.accepting_states else self._manager.top().eval({})
            )
            for (phi, loc) in self.states
        }

    @property
    def states(self) -> Collection[Q]:
        return self.expr_var_map.keys()

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
        dist_attr: Optional[str] = None,
    ) -> "StrelAutomaton":
        """Convert a STREL expression to an AFA with the given alphabet"""

        visitor = _ExprMapper(label_fn, polynomial, max_locs, dist_attr)
        visitor.visit(phi)

        aut = cls(phi, visitor._transitions, visitor.expr_var_map, visitor.var_node_map)

        return aut


class _ExprMapper(Generic[K]):
    """Post-order visitor for creating transitions"""

    def __init__(
        self,
        label_fn: LabellingFn[K],
        polynomial: Poly[K],
        max_locs: int,
        dist_attr: Optional[str] = None,
    ) -> None:
        assert max_locs > 0, "STREL graphs should have at least 1 location"
        self.max_locs = max_locs
        self.dist_attr = dist_attr or "weight"
        # Maps the string representation of a subformula to the AFA node
        # This is also the visited states.
        self.expr_var_map: dict[Q, Poly[K]] = dict()
        # Map from the polynomial var string to the state in Q
        self.var_node_map: dict[str, Q] = dict()
        # Maps the transition relation
        self._transitions = Transitions(dict(), polynomial, label_fn)
        self.transitions = self._transitions.mapping
        # Create a const polynomial for tracking nodes
        self.manager = self._transitions.manager

    def _add_aut_state(self, phi: strel.Expr) -> None:
        # phi_str = str(phi)
        # not_phi = ~phi
        # not_phi_str = str(not_phi)
        # for loc in range(self.max_locs):
        #     self.expr_var_map.setdefault(
        #         (phi, loc),
        #         self.manager.declare(str((phi_str, loc))),
        #     )
        #     self.var_node_map.setdefault(str((phi_str, loc)), (phi, loc))
        #     self.expr_var_map.setdefault(
        #         (not_phi, loc),
        #         self.manager.declare(str((not_phi_str, loc))),
        #     )
        #     self.var_node_map.setdefault(str((not_phi_str, loc)), (not_phi, loc))
        pass

    def _add_expr_alias(self, phi: strel.Expr, alias: strel.Expr) -> None:
        phi_str = str(phi)
        # not_phi = ~phi
        # not_phi_str = str(not_phi)
        for loc in range(self.max_locs):
            self.expr_var_map[(phi, loc)] = self.expr_var_map[(alias, loc)]
            self.transitions[(phi, loc)] = self.transitions[(alias, loc)]
            self.var_node_map.setdefault(str((phi_str, loc)), (phi, loc))
            # self.expr_var_map[(~phi, loc)] = self.expr_var_map[(~alias, loc)]
            # self.transitions[(~phi, loc)] = self.transitions[(~alias, loc)]
            # self.var_node_map.setdefault(str((not_phi_str, loc)), (not_phi, loc))

    def _add_transition(self, phi: strel.Expr, transition: Callable[[Location, Alph], Poly[K]]) -> None:
        phi_str = str(phi)
        for loc in range(self.max_locs):
            self.expr_var_map.setdefault(
                (phi, loc),
                self.manager.declare(str((phi_str, loc))),
            )
            self.var_node_map.setdefault(str((phi_str, loc)), (phi, loc))
            self.transitions.setdefault((phi, loc), partial(transition, loc))

    def _expand_add_next(self, phi: strel.NextOp) -> None:
        if phi.steps is None:
            steps = 1
        else:
            steps = phi.steps
        # print(f"{steps=}")
        # Expand the formula into nested nexts
        for i in range(steps, 0, -1):
            # print(f"{i=}")
            expr: strel.Expr = strel.NextOp(i, phi.arg)
            self._add_aut_state(expr)

        for i in range(steps, 1, -1):
            # print(f"{i=}")
            expr = strel.NextOp(i, phi.arg)
            # Expand as X[t] arg = XX[t - 1] arg
            sub_expr = strel.NextOp(i - 1, phi.arg)
            self._add_transition(expr, lambda loc, _, sub_expr=sub_expr: self.expr_var_map[(sub_expr, loc)])
        # Add the final bit where there is no nested next
        # Expand as X[1] arg = X arg
        self._add_transition(strel.NextOp(1, phi.arg), lambda loc, _, arg=phi.arg: self.expr_var_map[(arg, loc)])

    def _expand_add_globally(self, phi: strel.GloballyOp) -> None:
        # G[a,b] phi = ~F[a,b] ~phi
        expr: strel.Expr = ~strel.EventuallyOp(phi.interval, ~phi.arg)
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
                self._add_transition(
                    phi,
                    lambda loc, alph: self._transitions(alph, (phi.arg, loc)) + self.expr_var_map[(phi, loc)],
                )
            case strel.TimeInterval(0 | None, int(t2)):
                # phi = F[0, t2] arg
                for i in range(t2, 0, -1):
                    expr: strel.Expr = strel.EventuallyOp(strel.TimeInterval(0, i), phi.arg)
                    self._add_aut_state(expr)
                for i in range(t2, 0, -1):
                    expr: strel.Expr = strel.EventuallyOp(strel.TimeInterval(0, i), phi.arg)
                    sub_expr: strel.Expr  # keeps track of the RHS of the OR operation in the expansion
                    if i > 1:
                        # Expand as F[0, t2] arg = arg | X F[0, t2-1] arg
                        sub_expr = strel.EventuallyOp(strel.TimeInterval(0, i - 1), phi.arg)
                    else:  # i == 1
                        # Expand as F[0, 1] arg = arg | X arg
                        sub_expr = phi.arg
                    self._add_transition(
                        expr,
                        lambda loc, alph, sub_expr=sub_expr: self._transitions(alph, (phi.arg, loc))
                        + self.expr_var_map[(sub_expr, loc)],
                    )

            case strel.TimeInterval(int(t1), None):
                # phi = F[t1,] arg = X[t1] F arg
                expr: strel.Expr = strel.NextOp(t1, strel.EventuallyOp(None, phi.arg))
                self.visit(expr)
                self._add_expr_alias(phi, expr)

            case strel.TimeInterval(int(t1), int(t2)):
                # phi = F[t1, t2] arg = X[t1] F[0, t2 - t1] arg
                expr: strel.Expr = strel.NextOp(
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
                self._add_transition(
                    phi,
                    lambda loc, alph: self._transitions(alph, (phi.rhs, loc))
                    + (self._transitions(alph, (phi.lhs, loc)) * self.expr_var_map[(phi, loc)]),
                )
            case strel.TimeInterval(int(t1), None):
                # phi = lhs U[t1,] rhs = ~F[0,t1] ~(lhs U rhs)
                expr: strel.Expr = ~strel.EventuallyOp(
                    strel.TimeInterval(0, t1),
                    ~strel.UntilOp(phi.lhs, None, phi.rhs),
                )
                self.visit(expr)
                self._add_expr_alias(phi, expr)
            case strel.TimeInterval(int(t1), int()):
                # phi = lhs U[t1,t2] rhs = (F[t1,t2] rhs) & (lhs U[t1,] rhs)
                expr: strel.Expr = strel.AndOp(
                    strel.EventuallyOp(phi.interval, phi.rhs),
                    strel.UntilOp(
                        interval=strel.TimeInterval(t1, None),
                        lhs=phi.lhs,
                        rhs=phi.rhs,
                    ),
                )
                self.visit(expr)
                self._add_expr_alias(phi, expr)

    def _expand_add_reach(self, phi: strel.ReachOp) -> None:
        d1 = phi.interval.start or 0.0
        d2 = phi.interval.end or math.inf

        def check_reach(loc: Location, input: Alph, d1: float, d2: float, dist_attr: str) -> Poly[K]:
            # use a modified version of networkx's all_simple_paths algorithm to generate all simple paths
            # constrained by the distance intervals.
            # Then, make the symbolic expressions for each path, with the terminal one being for the rhs
            expr = self.manager.bottom()
            for edge_path in _all_reach_edge_paths(input, loc, d1, d2, dist_attr):
                path = [loc] + [e[1] for e in edge_path]
                # print(f"{path=}")
                # Path expr checks if last node satisfies rhs and all others satisfy lhs
                path_expr = self.expr_var_map[(phi.rhs, path[-1])]
                for l_p in reversed(path[:-1]):
                    path_expr *= self.expr_var_map[(phi.lhs, l_p)]
                expr += path_expr
                # Break early if TOP/True
                if expr.is_top():
                    return expr
            return expr

        self._add_transition(phi, partial(check_reach, d1=d1, d2=d2, dist_attr=self.dist_attr))

    def _expand_add_somewhere(self, phi: strel.SomewhereOp) -> None:
        # phi = somewhere[d1,d2] arg = true R[d1,d2] arg
        expr: strel.Expr = strel.ReachOp(strel.true, phi.interval, phi.arg)
        self.visit(expr)
        self._add_expr_alias(phi, expr)

    def _expand_add_everywhere(self, phi: strel.EverywhereOp) -> None:
        # phi = everywhere[d1,d2] arg = ~ somewhere[d1,d2] ~arg
        expr: strel.Expr = ~strel.SomewhereOp(phi.interval, ~phi.arg)
        self.visit(expr)
        self._add_expr_alias(phi, expr)

    def _expand_add_escape(self, phi: strel.EscapeOp) -> None:
        def instantaneous_escape(loc: Location, input: Alph) -> Poly[K]:
            pass

        self._add_transition(phi, instantaneous_escape)

    def visit(self, phi: strel.Expr) -> None:
        # Skip if phi already visited
        if str(phi) in self.expr_var_map.keys():
            return
        # 1. If phi is not a leaf expression visit its Expr children
        # 2. Add phi and ~phi as AFA nodes
        # 3. Add the transition for phi and ~phi
        match phi:
            case strel.Identifier():
                self._add_transition(phi, lambda loc, alph: self._transitions(alph, (phi, loc)))
            case strel.NotOp(arg):
                self.visit(arg)
                self._add_transition(
                    phi,
                    lambda loc, alph: self._transitions(alph, (arg, loc)).negate(),
                )
            case strel.AndOp(lhs, rhs):
                self.visit(lhs)
                self.visit(rhs)
                self._add_aut_state(phi)
                self._add_transition(
                    phi,
                    lambda loc, alph: self._transitions(alph, (lhs, loc)) * self._transitions(alph, (rhs, loc)),
                )
            case strel.OrOp(lhs, rhs):
                self.visit(lhs)
                self.visit(rhs)
                self._add_aut_state(phi)
                self._add_transition(
                    phi,
                    lambda loc, alph: self._transitions(alph, (lhs, loc)) + self._transitions(alph, (rhs, loc)),
                )
            case strel.EverywhereOp(_, arg):
                self.visit(arg)
                self._add_aut_state(phi)
                self._expand_add_everywhere(phi)
                raise NotImplementedError()
            case strel.SomewhereOp(_, arg):
                self.visit(arg)
                self._add_aut_state(phi)
                self._expand_add_somewhere(phi)
            case strel.EscapeOp(_, arg):
                self.visit(arg)
                self._add_aut_state(phi)
                # self._expand_add_escape(phi)
                raise NotImplementedError("We currently don't support escape")
            case strel.ReachOp(lhs, _, rhs):
                self.visit(lhs)
                self.visit(rhs)
                self._add_aut_state(phi)
                self._expand_add_reach(phi)
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


def _all_reach_edge_paths(
    graph: Alph, loc: Location, d1: float, d2: float, dist_attr: str
) -> Iterator[list[tuple[Location, Location, float]]]:
    """Return all edge paths for reachable nodes. The path lengths are always between `d1` and `d2` (inclusive)"""

    # This adapts networkx's all_simple_edge_paths code.
    #
    # Citations:
    #
    # 1. https://xlinux.nist.gov/dads/HTML/allSimplePaths.html
    # 2. https://networkx.org/documentation/stable/_modules/networkx/algorithms/simple_paths.html#all_simple_paths
    def get_edges(node: Location) -> Iterable[tuple[Location, Location, float]]:
        return graph.edges(node, data=dist_attr, default=1.0)

    # The current_path is a dictionary that maps nodes in the path to the edge that was
    # used to enter that node (instead of a list of edges) because we want both a fast
    # membership test for nodes in the path and the preservation of insertion order.
    # Edit: It also keeps track of the cumulative distance of the path.
    current_path: dict[Location | None, None | tuple[None | Location, Location, float]] = {None: None}

    # We simulate recursion with a stack, keeping the current path being explored
    # and the outgoing edge iterators at each point in the stack.
    # To avoid unnecessary checks, the loop is structured in a way such that a path
    # is considered for yielding only after a new node/edge is added.
    # We bootstrap the search by adding a dummy iterator to the stack that only yields
    # a dummy edge to source (so that the trivial path has a chance of being included).
    stack: deque[Iterator[tuple[None | Location, Location, float]]] = deque([iter([(None, loc, 0.0)])])

    # Note that the target is every other reachable node in the graph.
    targets = graph.nodes

    while len(stack) > 0:
        # 1. Try to extend the current path.
        #
        # Checks if node already visited.
        next_edge = next((e for e in stack[-1] if e[1] not in current_path), None)
        if next_edge is None:
            # All edges of the last node in the current path have been explored.
            stack.pop()
            current_path.popitem()
            continue
        previous_node, next_node, next_dist = next_edge

        if previous_node is not None:
            assert current_path[previous_node] is not None
            prev_path_len = (current_path[previous_node] or (None, None, 0.0))[2]
            new_path_len = prev_path_len + next_dist
        else:
            new_path_len = 0.0

        # 2. Check if we've reached a target (if adding the next_edge puts us in the distance range).
        if d1 <= new_path_len <= d2:
            # Yield the current path, removing the initial dummy edges [None, (None, source)]
            ret: list[tuple[Location, Location, float]] = (list(current_path.values()) + [next_edge])[2:]  # type: ignore
            yield ret

        # 3. Only expand the search through the next node if it makes sense.
        #
        # Check if the current cumulative distance (using previous_node) + new_dist is in the range.
        # Also check if all targets are explored.
        if new_path_len <= d2 and (targets - current_path.keys() - {next_node}):
            # Change next_edge to contain the cumulative distance
            update_edge = next_edge[:-1] + (new_path_len,)
            current_path[next_node] = update_edge
            stack.append(iter(get_edges(next_node)))
            pass
