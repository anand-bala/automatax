______________________________________________________________________

## title: "Repeatibility Package: Monitoring Spatially Distributed Cyber-Physical Systems with Alternating Finite Automata"

# Alternating Automata for Monitoring STREL specifications

The `automatix.afa` module implements weighted alternating finite automata over
algebra defined in `automatix.algebra.semiring`.

The repeatibility package is located at
<https://github.com/anand-bala/automatix/tree/hscc2025re>

## Reproducing Results

In this package, we provide the code to generate the results columns for Table 1, along
with the data used to generate the trajectories in Figures 1 and 2.

For the figures, the images for each map is within the directory `./data/` in the form
of PNG files.
The CSV files with the same timestamp as the image files correspond to traces.

To run the experiments, you can use the below instructions or use Docker as follows:

```bash
docker run -it --rm ghcr.io/anand-bala/automatix:hscc2025re
```

## Using the project

If you are just using it as a library, the Git repository should be installable pretty
easily using

```bash
pip install https://github.com/anand-bala/automatix/archive/refs/heads/hscc2025re.tar.gz
```

If you want develop the project or run examples, you will need to install
[Pixi](https://pixi.sh/latest/).
Once you install it for you platform, you can install the project development
dependencies and the current project as an editable install using:

```bash
pixi install -e dev
```

Then, you can use the following command to activate the pixi environment:

```bash
pixi shell -e dev
```

From here, you can look into the `examples` folder for some examples, and generally hack
away at the code.

## Running examples

To use an example trace, from the directory `./examples/swarm-monitoring/` run:

```bash
python ./monitoring_example.py \
    --spec "<spec_file>.py" \
    --map "<map_file>.json" \
    --trace "<trace_file>.csv" \
    # --ego "drone_03" \ # Skip to monitor from all locations, or specify multiple times. \
    # --timeit # Do rigorous profiling \
    # --online # Do online monitoring \
```

- The files, `./establish_comms_spec.py` and `./reach_avoid_spec.py` , are specification
  files (can be used to substitute the `<spec_file>.py` field above.

- The directories in `data/` contain the map files (named `map_params.json`) and trace
  files (CSV files under each directory).

For example

```bash
python ./monitoring_example.py \
      --spec establish_comms_spec.py \
      --map data/Map_1/map_params.json \
      --trace data/Map_1/log_2024_11_14_09_53_09.csv \
      --timeit
```

To run all examples shown in the HSCC 2025 paper, run:

```bash
python ./run_hscc_experiments.py
```

### Generating more traces

The code in `data/swarm-monitoring/swarmlab/` is used to generate trajectories. The main
file is `swarmlab/example_simulation.m`, which can be edited to point to various maps
and drone configurations (see the bottom of the file). Each map file is one of the
`param_map_<num>.m`, which can be edited to make new maps.

Before running the simulations, ensure that you have MATLAB installed and the
[Swarmlab](https://github.com/lis-epfl/swarmlab) package added in the [MATLAB search
path](https://www.mathworks.com/help/matlab/search-path.html).
