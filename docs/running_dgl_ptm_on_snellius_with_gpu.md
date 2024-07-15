# Running DGL-PTM on GPU (snellius)

### Import required modules (if needed)
module load 2023 CUDA/12.1.1 cuDNN/8.9.2.26-CUDA-12.1.1 

### Create environment from file 
mamba env create -f environment.yaml
* This assumes that the user already has mamba installed on the machine. Conda can also be used but at the cost of speed.

### Test GPU run
python gpu_sample_run.py
* Note that access to a GPU node is required to run on GPU. This can be done on snellius using the following command:
salloc --ntasks=1 -t 2:00:00 -p gpu --gpus-per-node=1


## If environment.yaml file does not work, an alternative approach:
### Create empty mamba environment
mamba create -n dgl_ptm_gpu

### Install dependencies
mamba install python=3.11, numpy, pydantic>=2, scipy-1.10.1, xarray, zarr, pytorch=2.1.1, torchvision, torchaudio, pytorch-cuda=12.1 -c pytorch -c nvidia

### Install dgl from wheel
pip install  dgl -f https://data.dgl.ai/wheels/cu121/repo.html
* Note that DGL cannot be installed with mamba due to dependencies mismatch.

### Install dgl-ptm 
pip install -e ../dgl_ptm

### Test GPU run
python gpu_sample_run.py 
* Note that access to a GPU node is required to run on GPU. This can be done on snellius using the following command: 
salloc --ntasks=1 -t 2:00:00 -p gpu --gpus-per-node=1
