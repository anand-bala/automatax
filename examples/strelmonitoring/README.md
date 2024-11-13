# Monitoring with STREL Automata

To run the example for offline monitoring with Boolean semantics, run:

```bash
python ./offline_monitoring.py \
    --spec ./reach_avoid_spec.py \
    --map ./sample_traj_data/Map_1/whole_run/param_map_1.json \
    --trace ./sample_traj_data/Map_1/whole_run/param_map_1_log_2024_11_12_14_09_20.csv \
    --ego "drone_03" \ # Skip to monitor from all locations, or specify multiple times.
```


<!-- The maps go in parameters. --> 
<!-- The create_shifted_buildings go in graphics/graphics_map. -->
<!--  'example_oltafi_saber.m' is the main file. Create a logs directory. -->
