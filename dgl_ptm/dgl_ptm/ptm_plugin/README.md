# DGL-PTM: Plugin Branch

## Description
This plugin has been adapted from code used in generating data for the preliminary, subsidy, and efficacy experiments with the Deep Graph Library Poverty Trap Model (DGL-PTM). 

An existing poverty trap model [1] was completely reimplemented in a way that accommodated agent populations on the order of millions and additional functionalities by leveraging the message passing capabilities of Deep Graph Library [2]. Now that the generalized version of [DGL-ABM](https://github.com/SDCCA/dgl_abm), is nearing completion, this poverty trap model plugin serves as the first test case of the DGL-ABM pluggy capacity.

## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [Contact](#contact)
- [References](#references)


## Installation

### Prequisites
- DGL-ABM and its dependencies installed in your environment (see [DGL-ABM repository](https://github.com/SDCCA/dgl_abm))

```bash
# Clone the repository
git clone https://github.com/SDCCA/dgl_abm_ptm.git

# Install in the same environment as DGL-ABM
cd dgl_abm_ptm
python -m pip install .
```
## Usage
Note that the model can be run with or without GPU access. A test script is currently in development.

### Plugin Files
```
ptm_plugin
│   pyproject.toml
│   README.md
│   setup.py
│   __init__.py
│
├───agent (support functions for updating agent/node attributes)
│       capital_update.py
│       income_generation.py
│       wealth_consumption.py
│       __init__.py
│
├───agentInteraction (support functions for edge-based calculations)
│       exchange_capital.py
│       weight_update.py
│       __init__.py
│
└───extensions (pluggy hook implementations)
        agent_update_extension.py
        step_extension.py
        __init__.py
```

### Cluster-Specific Setup Tips
- Check the CUDA module loads in the run scripts and modify as needed to match the cluster in use.
- Set the conda location environment variable to use bash scripts.
```bash
export CONDA_HOME=/path/to/your/anaconda3  # Replace with your anaconda path
```

## Contributors and Acknowledgments
The model code forming the basis of this plugin was designed and developed with support from the Netherlands eScience Center by the Dutch Research Council (NWO) under contract 27020G08, titled “Computing societal dynamics of climate change adaptation in cities,” through the contributions of Meiert Grootes, Pranav Chandramouli, Sara Alidoost, and Victoria Garibay. Acknowledgements to Debraj Roy and Tatiana Filatova for consultation on the model design, Namitha Jopan for foundational work on the past model, and Thijs van Lankveld for contributions to later versions of the model.

## Contact
Victoria Garibay, Ph.D. - [Contact Form](https://vmgaribay.github.io/portfolio/contact_form.html) | [GitHub Profile](https://github.com/vmgaribay)

## References

[1] Namitha T. Joppan. "Modelling Poverty Alleviation Strategies Using Resilience Thinking." Master’s thesis, University of Amsterdam, 2021. [https://scripties.uba.uva.nl/search?id=record_30354](https://scripties.uba.uva.nl/search?id=record_30354)

[2] Minjie Wang, Da Zheng, Zihao Ye, Quan Gan, Mufei Li, Xiang Song, Jinjing Zhou, Chao Ma, Lingfan Yu, Yu Gai, Tianjun Xiao, Tong He, George Karypis, Jinyang Li, Zheng Zhang. "Deep Graph Library: A Graph-Centric, Highly-Performant Package for Graph Neural Networks." arXiv preprint arXiv:1909.01315, 2019. [https://arxiv.org/abs/1909.01315](https://arxiv.org/abs/1909.01315)