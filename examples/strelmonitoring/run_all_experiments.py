"""Run all experiments for HSCC 2025."""

import argparse
import itertools
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

CURRENT_DIR = Path(__file__).parent

EXPERIMENTS = [
    # Map, Trace
    (  # Map 1
        "./sample_traj_data/Map_1/whole_run/param_map_1.json",
        "./sample_traj_data/Map_1/whole_run/param_map_1_log_2024_11_12_14_09_20.csv",
    ),
    (  # Map 2
        "./sample_traj_data/Map_2/whole_run/param_map_2.json",
        "./sample_traj_data/Map_2/whole_run/param_map_2_log_2024_11_12_13_53_21.csv",
    ),
    (  # Map 3
        "./sample_traj_data/Map_3/whole_run/param_map_3.json",
        "./sample_traj_data/Map_3/run_2/param_map_3_log_2024_11_12_17_52_42.csv",
    ),
    (  # Map 4
        "./sample_traj_data/Map_4/whole_run/param_map_4.json",
        "./sample_traj_data/Map_4/param_map_4_log_2024_11_12_16_54_58.csv",
    ),
    (  # Map 5
        "./sample_traj_data/Map_5/whole_run/param_map_5.json",
        "./sample_traj_data/Map_5/run_1/param_map_5_log_2024_11_12_17_10_39.csv",
    ),
]

SPECS = [
    "./establish_comms_spec.py",
    "./reach_avoid_spec.py",
]


@dataclass
class Args:

    @classmethod
    def parse_args(cls) -> "Args":
        parser = argparse.ArgumentParser(description="Run all experiments for HSCC 2025")

        args = parser.parse_args()
        return Args(**vars(args))


def main(args: Args) -> None:
    for spec, (map_file, trace_file), online in itertools.product(
        map(lambda p: Path(CURRENT_DIR, p), SPECS),
        map(lambda p: (Path(CURRENT_DIR, p[0]), Path(CURRENT_DIR, p[1])), EXPERIMENTS),
        [False, True],
    ):
        experiment_script = [
            sys.executable,
            Path(CURRENT_DIR, "./monitoring_example.py"),
            "--timeit",
            "--spec",
            spec,
            "--map",
            map_file,
            "--trace",
            trace_file,
        ]
        if online:
            experiment_script.append("--online")
        subprocess.run(experiment_script)
    pass


if __name__ == "__main__":
    main(Args.parse_args())
