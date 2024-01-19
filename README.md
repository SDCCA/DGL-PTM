The deep graph library reimplementation of the poverty trap model.

# Installation:

## Create environment from file

mamba env create -f environment.yaml

* This assumes that the user already has mamba installed on the machine. Conda can also be used but at the cost of speed.

## If environment.yaml file does not work, an alternative approach:

### Create empty mamba environment

mamba create -n dgl_ptm

### Install dependencies

mamba install python=3.11 numpy scipy-1.10.1 xarray zarr pyyaml ipykernel pytorch=2.1.1 torchvision=0.16.0 torchaudio=2.1.0 dgl==1.1.3 -c pytorch -c dglteam

### Install dgl-ptm

pip install -e dgl_ptm
