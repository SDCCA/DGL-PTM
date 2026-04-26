#!/bin/bash

# Environment 
export CONDA_HOME="C:/Users/Victoria/miniconda3"
source ${CONDA_HOME}/etc/profile.d/conda.sh
conda activate dgl_ptm_gpu

export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:64


log_file="default_reconstruction_run.log"
> "$log_file" 

# Experimental setup

# Read seeds from default_reconstruction_seeds.txt (corrupted data from earlier experiments)
seeds=()
readarray -t seeds < <(cat AdditionalExperiments/default_reconstruction_seeds.txt | tr ',' '\n' | tr -s ' ' '\n')

total_runs=${#seeds[@]}
counter=0
restart=0
earlystop=3

echo "Script: default_reconstruction_run.sh"

for seed in "${seeds[@]}"
    do
        ((counter++))
        if [ "$counter" -ge "$restart" ] && [ "$counter" -le "$earlystop" ]; then
            date=$(date)
            start=$(date +%s)
            echo "$date Started run $counter/$total_runs with seed: $seed" | tee -a "$log_file"
            variation="--seed $seed --steps 75 --root_path D:/UvA-RD/2026ReviewResponse/output/Default/Reconstruction2/"
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

