"""
Specification for each drone saying:

> Avoid the obstacles, and until the drone reaches the goal, it should be able to communicate to a ground station within two hops.
"""

REACH_AVOID = r"""
(G ! obstacle) & ((somewhere[0,2] groundstation) U goal)
    """

SPECIFICATION = REACH_AVOID

DIST_ATTR = "hop"
