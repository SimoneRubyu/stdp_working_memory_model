#!/bin/bash

for mu in $(LC_NUMERIC=C seq 0.0 0.05 1.0); do
    
    data_path="data_${mu}"
    
    echo "====================================================="
    echo "Simulation: path $data_path, mu $mu"
    echo "====================================================="
    
    python3 run_model.py --path $data_path --mu $mu
    python3 analysis_stdp.py --path $data_path

done