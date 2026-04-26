#!/bin/bash


# Environment
source ${CONDA_HOME}/etc/profile.d/conda.sh
conda activate dgl_ptm_gpu

log_file="disruption_run_5f3.log"
> "$log_file" 

# Experimental setup

# Read seeds from seeds.txt
seeds=()
readarray -t seeds < <(cat seeds.txt | tr ',' '\n' | tr -s ' ' '\n')

total_runs=${#seeds[@]}
counter=0
restart=0
earlystop=30

echo "Script: disruption_run_5_f3.sh"

for seed in "${seeds[@]}"
    do
        ((counter++))
        if [ "$counter" -ge "$restart" ] && [ "$counter" -le "$earlystop" ]; then
            date=$(date)
            start=$(date +%s)
            echo "$date Started run $counter/$total_runs with seed: $seed" | tee -a "$log_file"
            variation="--seed $seed --steps 51 --agents 10000 --root_path D:/UvA-RD/2026ReviewResponse/output/Disruption/disrupt_5_f3 --shocks [0.5,3]"
            python gpu_default.py $variation 
            finish=$(date +%s)
            date=$(date)
            echo "$date Finished run $counter/$total_runs with seed: $seed" | tee -a "$log_file"
            elapsed=$(($finish-$start))
            echo "Elapsed time: $elapsed" | tee -a "$log_file"
        fi
    done


wait


echo " $date - Runs $restart through $earlystop have been attempted."

