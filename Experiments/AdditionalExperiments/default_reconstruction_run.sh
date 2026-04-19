#!/bin/bash
#SBATCH --job-name=defPTM
#SBATCH -p gpu
#SBATCH --gpus=1

#            d-hh:mm:ss
#SBATCH --time=05:00:00
module load 2023 
module load CUDA/12.1.1 
module load cuDNN/8.9.2.26-CUDA-12.1.1 

if ! mount | grep -q "/tmp/UvA-RD"; then
    echo "Mounting UvA-RD..."
    rclone -vv mount --use-cookies --timeout 24h UvA-RD:IVI-FNWI-328-DGL-PTM\ \(Projectfolder\) /tmp/UvA-RD --vfs-cache-mode full --daemon
else
    echo "/tmp/UvA-RD is already mounted."
fi

log_file="default_run_times.log"
> "$log_file" 

# Environment (Snellius specific)
source ${CONDA_PREFIX}/etc/profile.d/conda.sh
#conda env create -f ../environment.yml --name dgl_ptm_gpu
conda activate dgl_ptm_gpu

# Experimental setup

# Read seeds from seeds.txt
seeds=()
readarray -t seeds < <(cat default_reconstruction_seeds.txt | tr ',' '\n' | tr -s ' ' '\n')

total_runs=${#seeds[@]}
counter=0
restart=0
earlystop=3

echo "Script: default_run.sh"

for seed in "${seeds[@]}"
    do
        ((counter++))
        if [ "$counter" -ge "$restart" ] && [ "$counter" -le "$earlystop" ]; then
            date=$(date)
            start=$(date +%s)
            echo "$date Started run $counter/$total_runs with seed: $seed" | tee -a "$log_file"
            variation="--seed $seed --steps 75 --root_path output/default/reconstruction/"
            python -m cProfile -o "profile_run_$counter.prof" ../gpu_default.py $variation 
            finish=$(date +%s)
            date=$(date)
            echo "$date Finished run $counter/$total_runs with seed: $seed" | tee -a "$log_file"
            elapsed=$(($finish-$start))
            echo "Elapsed time: $elapsed" | tee -a "$log_file"
        fi
    done


wait


echo " $date - Runs $restart through $earlystop have been attempted."

