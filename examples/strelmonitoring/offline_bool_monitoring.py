import argparse
import csv
import itertools
import json
import math
from collections import deque
from pathlib import Path
from typing import Self, Sequence, TypeAlias

import networkx as nx
import numpy as np
from pydantic import BaseModel, Field, model_validator

from automatix.logic import strel

DRONE_COMMS_RADIUS: float = 40
GCS_COMMS_RADIUS: float = 60
CLOSE_TO_GOAL_THRESH = 0.5
CLOSE_TO_OBSTACLE_THRESH = 0.1

Location: TypeAlias = int | str
Alph: TypeAlias = "nx.Graph[Location]"


class GroundStation(BaseModel):
    id: int
    y: float = Field(alias="north")
    x: float = Field(alias="east")


class Building(BaseModel):
    id: int
    y: float = Field(alias="north")
    x: float = Field(alias="east")


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

    @classmethod
    def parse(cls) -> "Args":
        parser = argparse.ArgumentParser(
            description="Run offline monitoring example", formatter_class=argparse.ArgumentDefaultsHelpFormatter
        )
        for name, field in cls.model_fields.items():
            ty = field.annotation or str
            parser.add_argument(
                f"--{field.alias or name}",
                help=field.description,
                type=lambda arg, ty=ty: ty(arg),
                required=field.is_required(),
            )
            # print(f"{field.alias or name:10s} {field.annotation} {field.description} {field.is_required()}")

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
    trace: list[tuple[float, "nx.Graph[Location]"]], map_info: Map
) -> list[tuple[float, "nx.Graph[Location]"]]:
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
        # Unpack drone locations
        drone_centers, drone_ids = zip(
            *[
                ((dr_data["pos_x"], dr_data["pos_y"]), dr)
                for dr, dr_data in sample.nodes(data=True)
                if dr_data["kind"] == "drone"
            ]
        )
        drone_centers = np.array(drone_centers)
        drone_ids = list(drone_ids)
        assert drone_centers.shape == (len(drone_centers), 2)

        # Compute distance to obstacles
        dist_to_each_obstacle = cdist(drone_centers, obstacle_centers, "euclidean")
        assert dist_to_each_obstacle.shape == (len(drone_ids), len(obstacle_ids))
        # Compute min dist to obstacles (along each row)
        # Subtract the radius and max by 0
        min_dist_to_obstacle = np.maximum(
            np.absolute((np.amin(dist_to_each_obstacle, axis=1) - obstacle_radius)),
            np.array(0.0),
        )
        assert min_dist_to_obstacle.shape == (len(drone_ids),)

        # For each drone, add the dist_to_obstacle attribute
        for i, dr in enumerate(drone_ids):
            sample.nodes[dr]["dist_to_obstacle"] = min_dist_to_obstacle[i]
    return trace


def read_trace(trace_file: Path, map_info: Map) -> Sequence[tuple[float, nx.Graph]]:
    """Convert raw trace into dynamic graph signals"""
    # Read the trace file as a csv
    with open(trace_file, "r") as f:
        reader = csv.DictReader(f)
        raw_trace = [TraceSample.model_validate(row) for row in reader]

    # Convert the map info GCSs into a graph too
    gcs_graph: "nx.Graph[str]" = nx.Graph()
    for gcs in map_info.ground_stations:
        gcs_graph.add_node(f"gcs_{gcs.id:02d}", kind="groundstation", pos_x=gcs.x, pos_y=gcs.y)

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
        g.add_node(
            f"drone_{sample.drone:02d}", kind="drone", **sample.model_dump(include={"pos_x", "pos_y", "vel_x", "vel_y"})
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
    # Read the mapinfo file
    with open(args.map_info, "r") as f:
        map_info = Map.model_validate(json.load(f))
    trace = list(read_trace(args.trace, map_info))
    trace = _get_distance_to_obstacle(trace, map_info)


if __name__ == "__main__":
    main(Args.parse())
