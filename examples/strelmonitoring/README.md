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
