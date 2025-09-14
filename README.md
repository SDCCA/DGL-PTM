# DGL-PTM: Experiment Branch

## Description

Included is all supporting code used in generating data for the preliminary, subsidy, and efficacy experiments with the Deep Graph Library Poverty Trap Model (DGL-PTM). This branch of the repository ("Experiments") serves as an archive of the DGL-PTM as it was applied in a manuscript (pending submission details). 

An existing poverty trap model [1] was completely reimplemented in a way that accommodated agent populations on the order of millions and additional functionalities by leveraging the message passing capabilities of Deep Graph Library [2]. This is the preliminary phase and first practical implementation of [DGL-ABM](https://github.com/SDCCA/dgl_abm), a continuing project to develop a generalized framework for agent based modelling based on repurposed foundational elements from Deep Graph Library. 



## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [Contact](#contact)
- [References](#references)


## Installation

```bash
# Clone the repository
git clone https://github.com/SDCCA/DGL-PTM.git
cd DGL-PTM

# Create and activate conda environment
conda env create -f environment.yaml
conda activate dgl_ptm_gpu

# Check Installation
python -c "import dgl; print(dgl.__version__)"
```
## Usage
Note that the model can be run with or without GPU access.
The experiments were conducted using the GPU nodes on Snellius, the Dutch National Supercomputer and bash scripts would need to be modified to be compatible with a new system. 

### Overview of Relevant Files
#### Main Model Run Files (1,000,000 agents):
gpu_default.py — Default specification (social exchange and adaptation options active)
gpu_no_adapt.py — No Adaptation specification (social exchange active)
gpu_no_social.py — No Social specification (adaptation active)
gpu_null.py — Null specification (no social exchange or adaptation)
#### Experimental Model Run Files (10,000 agents):
default_schemeA.py — Default specification, subsidized adaptation
default_schemeB.py — Default specification, increased adaptation efficacy
default_schemeSQ.py — Default specification
no_social_schemeA.py — No Social specification, subsidized adaptation
no_social_schemeB.py — No Social specification, increased adaptation efficacy
no_social_schemeSQ.py — No Social specification
#### Analysis Files:
data_processor.py — Processes the data from the main runs
scheme_data_processor.py — Processes the data from the subsidy and efficacy experiments
disruption_data_processor.py — Processes the data from the scheduled disruption experiments
#### Supporting Scripts:
default_run.sh — Monte Carlo runs of gpu_default.py
no_adapt_run.sh — Monte Carlo runs of gpu_no_adapt.py
no_social_run.sh — Monte Carlo runs of gpu_no_social.py
null_run.sh — Monte Carlo runs of gpu_null.py
default_run_schemeA.sh — Monte Carlo runs of default_schemeA.py
default_run_schemeB.sh — Monte Carlo runs of default_schemeB.py
default_run_schemeSQ.sh — Monte Carlo runs of default_schemeSQ.py
no_social_run_schemeA.sh — Monte Carlo runs of no_social_schemeA.py
no_social_run_schemeB.sh — Monte Carlo runs of no_social_schemeB.py
no_social_run_schemeSQ.sh — Monte Carlo runs of no_social_schemeSQ.py
disruption_run_6.sh — Monte Carlo runs of gpu_default.py with scheduled shock of $\Theta$ = 0.6 every fifth timestep
disruption_run_7.sh — Monte Carlo runs of gpu_default.py with scheduled shock of $\Theta$ = 0.7 every fifth timestep
disruption_run_8.sh — Monte Carlo runs of gpu_default.py with scheduled shock of $\Theta$ = 0.8 every fifth timestep
PyRun.sh — runs a Python script specified as an argument (e.g., data_processor.py)

### Cluster-Specific Setup Tips
- Check the CUDA module loads in the run scripts and modify as needed to match the cluster in use.
- Set the conda location environment variable to use bash scripts.
```bash
export CONDA_HOME=/path/to/your/anaconda3  # Replace with your anaconda path
```

## Contributors and Acknowledgments
This model code was designed and developed with support from the Netherlands eScience Center by the Dutch Research Council (NWO) under contract 27020G08, titled “Computing societal dynamics of climate change adaptation in cities” through the contributions of Meiert Grootes, Pranav Chandramouli, Sara Alidoost, and Victoria Garibay. Acknowledgements to Debraj Roy and Tatiana Filatova for consultation on the model design, Namitha Jopan for foundational work on the past model, and Thijs van Lankveld for contributions to later versions of the model.

## Contact
Victoria Garibay, Ph.D. - [Contact Form](https://vmgaribay.github.io/portfolio/contact_form.html) | [GitHub Profile](https://github.com/vmgaribay)

## References

[1] Namitha T. Joppan. "Modelling Poverty Alleviation Strategies Using Resilience Thinking." Master’s thesis, University of Amsterdam, 2021. [https://scripties.uba.uva.nl/search?id=record_30354](https://scripties.uba.uva.nl/search?id=record_30354)

[2] Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma, Lingfan Yu, Yu Gai, Tianjun Xiao, Tong He, George Karypis, Jinyang Li, Zheng Zhang. "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks." arXiv preprint arXiv:1909.01315, 2019. [https://arxiv.org/abs/1909.01315](https://arxiv.org/abs/1909.01315)
