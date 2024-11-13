import argparse
import csv
import itertools
import json
import math
import timeit
from collections import deque
from pathlib import Path
from typing import MutableSequence, Self, Sequence, TypeAlias

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, model_validator

from automatix.afa.strel import make_bool_automaton
from automatix.logic import strel

DRONE_COMMS_RADIUS: float = 40
GCS_COMMS_RADIUS: float = 60
CLOSE_TO_GOAL_THRESH = 0.5
CLOSE_TO_OBSTACLE_THRESH = 0.1

Location: TypeAlias = int
Alph: TypeAlias = "nx.Graph[Location]"


class GroundStation(BaseModel):
    id: int
    y: float = Field(alias="north")
    x: float = Field(alias="east")


class Building(BaseModel):
    id: int
    y: float = Field(alias="north")
    x: float = Field(alias="east")


class GoalPosition(BaseModel):
    x: float
    y: float


class MapInfo(BaseModel):
    num_blocks: int = Field(alias="nb_blocks")
    street_width_perc: float
    building_width: float
    street_width: float
    radius_of_influence: float = GCS_COMMS_RADIUS


class Map(BaseModel):
    map_properties: MapInfo
    buildings: list[Building]
    ground_stations: list[GroundStation]
    goal_position: GoalPosition = Field(alias="goal_positions")


class TraceSample(BaseModel):
    drone: int = Field(alias="Drone")
    time: float = Field(alias="Time")
    pos_y: float = Field(alias="North")
    pos_x: float = Field(alias="East")
    pos_z: float = Field(alias="Altitude")
    vel_x: float = Field(alias="Vx")
    vel_y: float = Field(alias="Vy")


class Args(BaseModel):
    spec_file: Path = Field(description="Path to a file with a STREL specification (python file)", alias="spec")
    map_info: Path = Field(description="Path to map.json file", alias="map")
    trace: Path = Field(description="Path to trace.csv file")

    ego_loc: list[str] = Field(
        description="Names of the ego location. Format is '(drone|groundstation)_(0i)', where `i` is the index. Default: all locations",
        default_factory=list,
        alias="ego",
    )
    timeit: bool = Field(description="Record performance", default=False)
    forward_run: bool = Field(description="To the forward method", default=True)

    @classmethod
    def parse(cls) -> "Args":
        parser = argparse.ArgumentParser(
            description="Run offline monitoring example",
        )
        parser.add_argument(
            "--spec", help="Path to a file with a STREL specification (python file)", type=lambda arg: Path(arg), required=True
        )
        parser.add_argument("--map", help="Path to map.json file", type=lambda arg: Path(arg), required=True)
        parser.add_argument("--trace", help="Path to trace.csv file", required=True)
        parser.add_argument(
            "--ego",
            help="Name of the ego location. Format is '(drone|groundstation)_(0i)', where `i` is the index. Default: all locations",
            action="append",
            required=False,
        )
        parser.add_argument("--timeit", help="Record performance", action="store_true")
        # parser.add_argument("--forward", help="Record performance", action="store_true")

        args = parser.parse_args()
        return Args(**vars(args))

    @model_validator(mode="after")
    def _path_exists(self) -> Self:
        assert self.spec_file.is_file(), "Specification file does not exist"
        assert self.map_info.is_file(), "Map info file does not exist"
        assert self.trace.is_file(), "Trace file does not exist"
        return self


def read_spec_file(spec_file: Path) -> tuple[strel.Expr, str]:
    """Return the STREL expression defined in the file along with the DIST_ATTR distance attribute for the spec"""
    import importlib.util

    assert spec_file.is_file()

    specification_module_spec = importlib.util.spec_from_file_location("strel_spec", spec_file.absolute())
    assert specification_module_spec is not None
    specification_module = importlib.util.module_from_spec(specification_module_spec)
    assert specification_module_spec.loader is not None
    specification_module_spec.loader.exec_module(specification_module)

    expr = strel.parse(specification_module.SPECIFICATION)
    dist_attr: str = specification_module.DIST_ATTR

    return expr, dist_attr


def _get_distance_to_obstacle(
    trace: MutableSequence[tuple[float, "nx.Graph[str]"]], map_info: Map
) -> MutableSequence[tuple[float, "nx.Graph[str]"]]:
    """Iterate over the graphs and, to each drone location, add a "dist_to_obstacle" parameter"""
    from scipy.spatial.distance import cdist

    # Unpack obstacles
    obstacle_centers, obstacle_ids = zip(*[((obs.x, obs.y), obs.id) for obs in map_info.buildings])
    assert len(obstacle_centers) == len(obstacle_centers)
    obstacle_centers = np.array(obstacle_centers)
    obstacle_ids = list(obstacle_ids)
    assert obstacle_centers.shape == (len(obstacle_centers), 2)
    obstacle_radius = map_info.map_properties.building_width

    for _, sample in trace:
        # Unpack locations
        loc_centers, loc_ids = zip(
            *[
                ((dr_data["pos_x"], dr_data["pos_y"]), dr)
                for dr, dr_data in sample.nodes(data=True)
                if dr_data["kind"] == "drone"
            ]
        )
        loc_centers = np.array(loc_centers)
        loc_ids = list(loc_ids)
        assert loc_centers.shape == (len(loc_centers), 2)

        # Compute distance to obstacles
        dist_to_each_obstacle = cdist(loc_centers, obstacle_centers, "euclidean")
        assert dist_to_each_obstacle.shape == (len(loc_ids), len(obstacle_ids))
        # Compute min dist to obstacles (along each row)
        # Subtract the radius and max by 0
        min_dist_to_obstacle = np.maximum(
            np.absolute((np.amin(dist_to_each_obstacle, axis=1) - obstacle_radius)),
            np.array(0.0),
        )
        assert min_dist_to_obstacle.shape == (len(loc_ids),)

        # For each location, add the dist_to_obstacle attribute
        for i, dr in enumerate(loc_ids):
            sample.nodes[dr]["dist_to_obstacle"] = min_dist_to_obstacle[i]
    return trace


def read_trace(trace_file: Path, map_info: Map) -> Sequence[tuple[float, nx.Graph]]:
    """Convert raw trace into dynamic graph signals"""
    # Read the trace file as a csv
    with open(trace_file, "r") as f:
        reader = csv.DictReader(f)
        raw_trace = [TraceSample.model_validate(row) for row in reader]

    goal_pos = np.array([map_info.goal_position.x, map_info.goal_position.y], dtype=np.float64)

    # Convert the map info GCSs into a graph too
    gcs_graph: "nx.Graph[str]" = nx.Graph()
    for gcs in map_info.ground_stations:
        # Compute distance to goal.
        gcs_pos = np.array([gcs.x, gcs.y])
        dist_sq = np.sum(np.square(goal_pos - gcs_pos))
        gcs_graph.add_node(
            f"gcs_{gcs.id:02d}",
            kind="groundstation",
            pos_x=gcs.x,
            pos_y=gcs.y,
            dist_to_goal=np.sqrt(dist_sq),
            dist_to_obstacle=0.0,
        )

    for d1, d2 in itertools.combinations(gcs_graph.nodes, 2):
        x1, y1 = gcs_graph.nodes[d1]["pos_x"], gcs_graph.nodes[d1]["pos_y"]
        x2, y2 = gcs_graph.nodes[d2]["pos_x"], gcs_graph.nodes[d2]["pos_y"]
        dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
        if dist <= map_info.map_properties.radius_of_influence:
            gcs_graph.add_edge(
                d1,
                d2,
                hop=1,
                dist=dist,
            )

    # For each timestamp, create a nx.Graph
    prev_time: float | None = None
    trace: deque[tuple[float, "nx.Graph[str]"]] = deque()
    for sample in raw_trace:
        if prev_time is None or prev_time < sample.time:
            # Detected new time point. Create a new graph
            trace.append((sample.time, nx.Graph()))
            prev_time = sample.time
            pass
        # Add current row to last graph
        g = trace[-1][1]
        # Compute distance to goal.
        drone_pos = np.array([sample.pos_x, sample.pos_y])
        dist_sq = np.sum(np.square(goal_pos - drone_pos))
        g.add_node(
            f"drone_{sample.drone:02d}",
            kind="drone",
            pos_x=sample.pos_x,
            pos_y=sample.pos_y,
            dist_to_goal=np.sqrt(dist_sq),
        )

    # Add an edge between drones if they are within communication distance.
    for _, g in trace:
        # print(f"{t:.2f}: {g.nodes=}")
        for d1, d2 in itertools.combinations(g.nodes, 2):
            x1, x2 = g.nodes[d1]["pos_x"], g.nodes[d2]["pos_x"]
            y1, y2 = g.nodes[d1]["pos_y"], g.nodes[d2]["pos_y"]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist <= DRONE_COMMS_RADIUS:
                g.add_edge(d1, d2, hop=1, dist=dist)

        # Connect a GCS to a drone if in radius
        new_edges = []
        for gcs, drone in itertools.product(gcs_graph.nodes, g.nodes):
            x1, y1 = gcs_graph.nodes[gcs]["pos_x"], gcs_graph.nodes[gcs]["pos_y"]
            x2, y2 = g.nodes[drone]["pos_x"], g.nodes[drone]["pos_y"]
            dist = math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            if dist <= map_info.map_properties.radius_of_influence:
                new_edges.append((gcs, drone, dict(hop=1, dist=dist)))

        # Merge the gcs graph and the drone graph
        g.update(gcs_graph)
        g.add_edges_from(new_edges)

    return trace


def assign_bool_labels(input: Alph, loc: Location, pred: str) -> bool:  # noqa: N802
    match pred:
        case "drone" | "groundstation":
            return input.nodes[loc]["kind"] == pred
        case "obstacle":
            return input.nodes[loc]["dist_to_obstacle"] <= CLOSE_TO_OBSTACLE_THRESH
        case "goal":
            return input.nodes[loc]["dist_to_goal"] <= CLOSE_TO_GOAL_THRESH

    ret = input.nodes[loc][pred]
    assert isinstance(ret, bool)
    return ret


def main(args: Args) -> None:
    print("================================================================================")
    print(
        """
WARNING:
    Longer traces will take time to run due to pre-calculations needed to make
    the dynamic graphs. This does not measure the actual time it takes to
    monitor things (which will be timed and reported).

"""
    )
    spec, dist_attr = read_spec_file(args.spec_file)
    print(f"phi = {str(spec)}")
    print()
    with open(args.map_info, "r") as f:
        map_info = Map.model_validate(json.load(f))
    trace = list(read_trace(args.trace, map_info))
    print(f"Trace Length  = {len(trace)}")
    print()
    max_locs = max([g.number_of_nodes() for _, g in trace])
    print(f"Num Locations = {max_locs}")

    trace = _get_distance_to_obstacle(trace, map_info)
    # Remove timestamps from trace, and relabel the traces with integer nodes
    remapping = {name: i for i, name in enumerate(trace[0][1].nodes)}
    new_trace: list["nx.Graph[int]"] = [nx.relabel_nodes(g, remapping) for _, g in trace]  # type: ignore
    assert len(new_trace) == len(trace)
    assert isinstance(new_trace[0], nx.Graph)

    monitor = make_bool_automaton(
        spec,
        assign_bool_labels,
        max_locs,
        dist_attr,
    )
    final_mapping = monitor.final_mapping

    def forward_run() -> dict[str, bool]:
        states = {name: monitor.initial_at(loc) for name, loc in remapping.items()}
        for input in new_trace:
            states = {name: monitor.next(input, state) for name, state in states.items()}
        return {name: state.eval(final_mapping) for name, state in states.items()}

    if len(args.ego_loc) > 0:
        for ego_loc in map(lambda e: remapping[e], args.ego_loc):
            if args.timeit:
                print("Logging time taken to monitor trace.")
                timer = timeit.Timer(lambda ego_loc=ego_loc: monitor.check_run(ego_loc, new_trace), "gc.enable()")
                n_loops, time_taken = timer.autorange()
                print(f"Ran monitoring code {n_loops} times. Took {time_taken} ns")
            else:
                print(f"Begin monitoring trace for ego location: {args.ego_loc}")
                start_time = timeit.default_timer()
                check = monitor.check_run(ego_loc, new_trace)
                end_time = timeit.default_timer()
                print(f"Completed monitoring in: \t\t{end_time - start_time} nanoseconds")
                print()
                print(f"\tphi @ {args.ego_loc} = {check}")

    else:
        if args.timeit:
            print("Logging time taken to monitor trace.")
            timer = timeit.Timer(forward_run, "gc.enable()")
            n_loops, time_taken = timer.autorange()
            print(f"Ran monitoring code {n_loops} times. Took {time_taken} ns")
        else:
            print("Begin monitoring trace")
            start_time = timeit.default_timer()
            check = forward_run()
            end_time = timeit.default_timer()
            print(f"Completed monitoring in: \t\t{end_time - start_time} nanoseconds")
            print()
            for name, sat in check.items():
                print(f"\tphi @ {name} = {sat}")

    print("================================================================================")


if __name__ == "__main__":
    main(Args.parse())
