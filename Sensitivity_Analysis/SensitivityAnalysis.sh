#!/bin/bash
#SBATCH --job-name=SA1_seed1
#SBATCH -p gpu
#SBATCH --gpus=1

#              d-hh:mm:ss
#SBATCH --time=00:05:00


module load 2023
conda activate dgl_ptm_gpu

# Sensitivity Analysis


seed=1
counter=0
start=1
earlystop=3
sample_csv="SaltelliSampleParams-n1024.csv"

while IFS=, read -r RunID homophily local_ratio noise_ratio shock_alpha shock_beta; do    
    # Check range compliance
    if [ "$earlystop" != "None" ] && [ "$RunID" -gt "$earlystop" ]; then
        break
    fi
    # Check range compliance and print parameters
    if [ "$RunID" -ge "$start" ]; then
        echo Run: "$RunID"
        echo Homophily: "$homophily" Local Ratio: "$local_ratio" Noise Ratio: "$noise_ratio" Shock Alpha: "$shock_alpha" Shock Beta: "$shock_beta"
        python SensitivityAnalysis.py --seed "$seed" --run_id "$RunID" --homophily "$homophily" --local "$local_ratio" --noise "$noise_ratio" --shock_a "$shock_alpha" --shock_b "$shock_beta"


    fi

done < "$sample_csv"
wait
echo " $(date) - Runs $start through $earlystop are complete for seed $seed."


   



