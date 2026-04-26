# DGL-PTM: Additional Experiments Branch

## Description

Included is all supporting code used in generating data with the Deep Graph Library Poverty Trap Model (DGL-PTM) for the review response regarding the JOSCI article "Pricing Adaptation: A Simulated Study of Poverty Intervention Strategies Under Climate Shocks" by V.M.Garibay & D. Roy [citation pending]. This branch of the repository ("additional-experiments") was branched from "experiments" which is an archive of the DGL-PTM as it was applied in the originally submitted manuscript. Please see that branch for more details on the model and its original application, found [here](https://github.com/SDCCA/dgl_abm/tree/experiments). No changes were made to the model functionalities, but some very minor edits were made to `wealth_consumption.py` and `utils.py` to accommodate running on a local machine versus the cluster.

Files pertaining to the additional experiments conducted in respnse to the review are available in the [Harvard Dataverse](https://doi.org/10.7910/DVN/SO1BZW). Data and summaries generated from original model runs used in the sensitivity analyisis are similarly available in the [original Harvard Dataverse dataset](https://doi.org/10.7910/DVN/H2IBDM).


## Table of Contents

- [Installation](#installation)
- [Usage](#usage)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [Contact](#contact)
- [References](#references)


## Installation
 Please see the [experiments branch](https://github.com/SDCCA/dgl_abm/tree/experiments) for details. 

## Usage
Please see the [experiments branch](https://github.com/SDCCA/dgl_abm/tree/experiments) for details on general usage.
These additional experiments were conducted locally on GPU using the included bash scripts. For the record, a full (inefficient) dump of the environment used for the local runs is included in the root directory of this branch as `2026ReviewEnvironment.yml`.

### Overview of Relevant Files

#### Analysis Files:
nnbellman_stratification.py — Generates and processes the data from the neural network equation replacement
agent_sensitivity.py — Generates PAWN analysis based on 25 1,000,000 agent runs from the original dataset
frequent_severe_disruption.py — Processes the data from the scheduled disruption experiments  

#### Supporting Scripts:
AdditionalExperiments/default_reconstruction_run.sh — Monte Carlo runs of gpu_default.py  
AdditionalExperiments/disruption_run_5f3.sh — Monte Carlo runs of gpu_default.py with scheduled shock of $\Theta$ = 0.5 every third timestep  
AdditionalExperiments/disruption_run_4f3.sh — Monte Carlo runs of gpu_default.py with scheduled shock of $\Theta$ = 0.4 every third timestep

#### Miscellenous:
AdditionalExperiments/default_reconstruction_seeds.txt — Seeds used in regeneration of missing data
AdditionalExperiments/2026ReviewEnvironment.yml — Dump of the local environment used for the additional experiments


## Contributors and Acknowledgments
This model code was designed and developed with support from the Netherlands eScience Center by the Dutch Research Council (NWO) under contract 27020G08, titled “Computing societal dynamics of climate change adaptation in cities” through the contributions of Meiert Grootes, Pranav Chandramouli, Sara Alidoost, and Victoria Garibay. Acknowledgements to Debraj Roy and Tatiana Filatova for consultation on the model design, Namitha Jopan for foundational work on the past model, and Thijs van Lankveld for contributions to later versions of the model.

## Contact
Victoria Garibay, Ph.D. - [Contact Form](https://vmgaribay.github.io/portfolio/contact_form.html) | [GitHub Profile](https://github.com/vmgaribay)

