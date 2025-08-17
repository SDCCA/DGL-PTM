#!/bin/bash
#SBATCH --job-name=defB
#SBATCH -p gpu
#SBATCH --gpus=1

#            d-hh:mm:ss
#SBATCH --time=10:00:00
module load 2023 
module load CUDA/12.1.1 
module load cuDNN/8.9.2.26-CUDA-12.1.1 

log_file="default_schemeB_run_times.log"
> "$log_file" 

# Environment (Snellius specific)
source ${CONDA_HOME}/etc/profile.d/conda.sh
#conda env create -f ../environment.yml --name dgl_ptm_gpu
conda activate dgl_ptm_gpu

# Experimental setup

# Read seeds from seeds.txt
seeds=()
readarray -t seeds < <(cat seeds.txt | tr ',' '\n' | tr -s ' ' '\n')

total_runs=${#seeds[@]}
counter=0
restart=0
earlystop=50

echo "Script: default_run_schemeB.sh"

for seed in "${seeds[@]}"
    do
        ((counter++))
        if [ "$counter" -ge "$restart" ] && [ "$counter" -le "$earlystop" ]; then
            date=$(date)
            start=$(date +%s)
            echo "$date Started run $counter/$total_runs with seed: $seed" | tee -a "$log_file"
            variation="--seed $seed --steps 51 --agents 10000 --root_path output/SchemeB/default"
            python gpu_default_schemeB.py $variation 
            finish=$(date +%s)
            date=$(date)
            echo "$date Finished run $counter/$total_runs with seed: $seed" | tee -a "$log_file"
            elapsed=$(($finish-$start))
            echo "Elapsed time: $elapsed" | tee -a "$log_file"
        fi
    done


wait


echo " $date - Runs $restart through $earlystop have been attempted."

