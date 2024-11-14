# Monitoring with STREL Automata

To run the example for offline monitoring with Boolean semantics, run:

```bash
python ./monitoring_example.py \
    --spec "<spec_file>.py" \
    --map "<map_file>.json" \
    --trace "<trace_file>.csv" \
    # --ego "drone_03" \ # Skip to monitor from all locations, or specify multiple times. \
    # --timeit # Do rigorous profiling \
    # --online # Do online monitoring \
```


<!-- The maps go in parameters. --> 
<!-- The create_shifted_buildings go in graphics/graphics_map. -->
<!--  'example_oltafi_saber.m' is the main file. Create a logs directory. -->


# Commands for both specs for maps 1-5

---------------------Map 1-------------------------

python ./monitoring_example.py \
    --spec "establish_comms_spec.py" \
    --map "sample_traj_data/Map_1/whole_run/param_map_1.json" \
    --trace "sample_traj_data/Map_1/whole_run/param_map_1_log_2024_11_12_14_09_20.csv" \
     --timeit 

python ./monitoring_example.py \
    --spec "reach_avoid_spec.py" \
    --map "sample_traj_data/Map_1/whole_run/param_map_1.json" \
    --trace "sample_traj_data/Map_1/whole_run/param_map_1_log_2024_11_12_14_09_20.csv" \
     --timeit 

---------------------Map 2-------------------------


python ./monitoring_example.py \
    --spec "establish_comms_spec.py" \
    --map "sample_traj_data/Map_2/whole_run/param_map_2.json" \
    --trace "sample_traj_data/Map_2/whole_run/param_map_2_log_2024_11_12_13_53_21.csv" \
     --timeit   
     
python ./monitoring_example.py \
    --spec "reach_avoid_spec.py" \
    --map "sample_traj_data/Map_2/whole_run/param_map_2.json" \
    --trace "sample_traj_data/Map_2/whole_run/param_map_2_log_2024_11_12_13_53_21.csv" \
     --timeit 

---------------------Map 3-------------------------


python ./monitoring_example.py \
    --spec "establish_comms_spec.py" \
    --map "sample_traj_data/Map_3/run_2/param_map_3_2024_11_12_17_51_14.json" \
    --trace "sample_traj_data/Map_3/run_2/param_map_3_log_2024_11_12_17_52_42.csv" \
     --timeit 

python ./monitoring_example.py \
    --spec "reach_avoid_spec.py" \
    --map "sample_traj_data/Map_3/run_2/param_map_3_2024_11_12_17_51_14.json" \
    --trace "sample_traj_data/Map_3/run_2/param_map_3_log_2024_11_12_17_52_42.csv" \
     --timeit 

---------------------Map 4-------------------------

python ./monitoring_example.py \
    --spec "establish_comms_spec.py" \
    --map "sample_traj_data/Map_4/param_map_4_2024_11_12_16_54_01.json" \
    --trace "sample_traj_data/Map_4/param_map_4_log_2024_11_12_16_54_58.csv" \
     --timeit 

python ./monitoring_example.py \
    --spec "reach_avoid_spec.py" \
    --map "sample_traj_data/Map_4/param_map_4_2024_11_12_16_54_01.json" \
    --trace "sample_traj_data/Map_4/param_map_4_log_2024_11_12_16_54_58.csv" \
     --timeit 


---------------------Map 5-------------------------

python ./monitoring_example.py \
    --spec "establish_comms_spec.py" \
    --map "sample_traj_data/Map_5/run_1/param_map_5_2024_11_12_17_09_33.json" \
    --trace "sample_traj_data/Map_5/run_1/param_map_5_log_2024_11_12_17_10_39.csv" \
     --timeit 

python ./monitoring_example.py \
    --spec "reach_avoid_spec.py" \
    --map "sample_traj_data/Map_5/run_1/param_map_5_2024_11_12_17_09_33.json" \
    --trace "sample_traj_data/Map_5/run_1/param_map_5_log_2024_11_12_17_10_39.csv" \
     --timeit 
