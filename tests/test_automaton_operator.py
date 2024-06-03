from typing import final

import equinox as eqx
import jax.nn
import jax.numpy as jnp
from jaxtyping import Array, Num, Scalar
from typing_extensions import TypeAlias

from automatax import Predicate
from automatax.automaton import Automaton, make_automaton_operator
from automatax.semirings import MaxPlusSemiring

Box: TypeAlias = Num[Array, " 4"]
Circle: TypeAlias = Num[Array, " 3"]
Point: TypeAlias = Num[Array, " 2"]


# Obstacle locations
RED_BOX: Box = jnp.array([0.0, 0.9, -1.0, -0.5])
"""red box in bottom right corner in the format `[x1, x2, y1, y2]`"""
GREEN_BOX: Box = jnp.array([0.2, 0.7, 0.8, 1.2])
"""green box in top right corner in the format `[x1, x2, y1, y2]`"""
ORANGE_BOX: Box = jnp.array([-1.0, -0.7, -0.2, 0.5])
"""orange box on the left in the format `[x1, x2, y1, y2]`"""

BLUE_CIRCLE: Circle = jnp.array([0.0, 0.0, 0.4])
"""blue circle in the center in the format `[x, y, radius]`"""


def signed_dist_box(point: Point, box: Box) -> Scalar:
    """Get the signed distance from a point to a box.

    If positive, the point is outside the box; otherwise, it is within the box.
    """
    # See: https://stackoverflow.com/a/30545544

    # Get the signed distance to the borders
    bottom_left = box[jnp.array([0, 2])] - point
    top_right = point - box[jnp.array([1, 3])]
    # Get the signed distance to the _closest_ border
    closest_border = jnp.maximum(bottom_left, top_right)
    dist = jnp.sqrt(jnp.sum(jax.nn.relu(closest_border) ** 2, axis=-1)) + jax.nn.relu(-jnp.amax(closest_border, axis=-1))

    # dist = jnp.linalg.vector_norm(jax.nn.relu(closest_border), axis=-1) + jax.nn.relu(-jnp.amax(closest_border, axis=-1))
    return dist


def test_signed_dist_to_box() -> None:
    point = jnp.array([-1.0, -0.75])
    dist = signed_dist_box(point, RED_BOX)

    assert dist.shape == ()
    assert jnp.allclose(dist, 1.0)
    print(dist)

    points = jnp.repeat(jnp.expand_dims(point, 0), 20, axis=0)
    assert points.shape == (20, 2)
    dists = jax.vmap(signed_dist_box, (0, None), 0)(points, RED_BOX)
    assert dists.shape == (20,)
    assert jnp.allclose(dists, 1.0)
    print(dists)


@final
class InRed(Predicate):
    def is_true(self, x: Num[Array, " n"]) -> bool:
        return jnp.all(signed_dist_box(x, RED_BOX) <= 0.0)

    def weight(self, x: Num[Array, " n"]) -> Scalar:
        return -jax.nn.relu(signed_dist_box(x, RED_BOX))


@final
class OutRed(Predicate):
    def is_true(self, x: Num[Array, " n"]) -> bool:
        return jnp.all(signed_dist_box(x, RED_BOX) > 0.0)

    def weight(self, x: Num[Array, " n"]) -> Scalar:
        return -jax.nn.relu(-signed_dist_box(x, RED_BOX))


@final
class InOrange(Predicate):
    def is_true(self, x: Num[Array, " n"]) -> bool:
        return jnp.all(signed_dist_box(x, ORANGE_BOX) <= 0.0)

    def weight(self, x: Num[Array, " n"]) -> Scalar:
        return -jax.nn.relu(signed_dist_box(x, ORANGE_BOX))


@final
class OutOrange(Predicate):
    def is_true(self, x: Num[Array, " n"]) -> bool:
        return jnp.all(signed_dist_box(x, ORANGE_BOX) > 0.0)

    def weight(self, x: Num[Array, " n"]) -> Scalar:
        return -jax.nn.relu(-signed_dist_box(x, ORANGE_BOX))


@final
class InGreen(Predicate):
    def is_true(self, x: Num[Array, " n"]) -> bool:
        return jnp.all(signed_dist_box(x, GREEN_BOX) <= 0.0)

    def weight(self, x: Num[Array, " n"]) -> Scalar:
        return -jax.nn.relu(signed_dist_box(x, GREEN_BOX))


@final
class OutGreen(Predicate):
    def is_true(self, x: Num[Array, " n"]) -> bool:
        return jnp.all(signed_dist_box(x, GREEN_BOX) > 0.0)

    def weight(self, x: Num[Array, " n"]) -> Scalar:
        return -jax.nn.relu(-signed_dist_box(x, GREEN_BOX))


@final
class Tautology(Predicate):
    def is_true(self, x: Num[Array, " n"]) -> bool:
        return True

    def weight(self, x: Num[Array, " n"]) -> Scalar:
        return jnp.array(0.0)


def test_weight_fn() -> None:
    sequential_aut = Automaton()
    sequential_aut.add_location(0, initial=True)
    sequential_aut.add_location(1)
    sequential_aut.add_location(2)
    sequential_aut.add_location(3, final=True)

    sequential_aut.add_transition(0, 0, guard=OutRed())
    sequential_aut.add_transition(0, 1, guard=InRed())
    sequential_aut.add_transition(1, 1, guard=OutGreen())
    sequential_aut.add_transition(1, 2, guard=InGreen())
    sequential_aut.add_transition(2, 2, guard=OutOrange())
    sequential_aut.add_transition(2, 3, guard=InOrange())
    sequential_aut.add_transition(3, 3, guard=Tautology())

    operator = make_automaton_operator(sequential_aut, MaxPlusSemiring)

    assert operator.initial_weights.shape == (4,)
    assert operator.initial_weights[0] == MaxPlusSemiring.ones(1).item()
    assert operator.final_weights.shape == (4,)
    assert operator.final_weights[3] == MaxPlusSemiring.ones(1).item()

    transitions = jax.jit(operator.cost_transitions)
    # transitions = operator.cost_transitions

    n_timesteps = 1500

    # We will generate a trajectory of a circle of radius 0.75 starting at theta = -pi/2 to pi
    angles = jnp.linspace(-jnp.pi / 2, jnp.pi * 3 / 2, n_timesteps)
    xs = jnp.cos(angles)
    ys = jnp.sin(angles)
    trajectory = jnp.stack((xs, ys), axis=0).T
    assert trajectory.shape == (n_timesteps, 2)

    assert transitions(trajectory[0]).shape == (4, 4)

    deltas = jax.vmap(transitions)(trajectory)
    assert deltas.shape == (n_timesteps, 4, 4)

    weights, _ = jax.lax.scan(
        lambda x, y: (MaxPlusSemiring.matmul(x, y), None), operator.initial_weights.reshape(1, -1), deltas
    )
    weight = MaxPlusSemiring.vdot(weights.squeeze(), operator.final_weights)
    assert weight.size == 1


if __name__ == "__main__":
    test_weight_fn()
