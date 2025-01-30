"""
Specification for each drone saying:

> If at any time the ego drone isn't within 1-2 hops of another drone, it must within 100 time steps (1 second) reestablish
connection with the flock or with a ground station.

"""

ESTABLISH_COMMS = r"""
G( (somewhere[1,2] drone) | (F[0, 100] somewhere[1,2] (drone | groundstation)) )
"""

SPECIFICATION = ESTABLISH_COMMS

DIST_ATTR = "hop"
