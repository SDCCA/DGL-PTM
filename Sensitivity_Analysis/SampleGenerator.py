# Sobol'/Saltelli Sample generation for sensitivity analysis...
from SALib.sample import sobol
import pandas as pd
import numpy as np


# Generate Samples

n=1024
#2d+2=12

# homophily   0-10
# local_ratio 0.25+/-0.05
# noise_ratio 0.05+/-0.01
# based on beta_distribution.ipynb
# event_alpha 4.13+/-50%
# event_beta  0.07+/-50%

problem={   "names": ["homophily","local_ratio","noise_ratio","shock_alpha","shock_beta"], 
            "num_vars":5,
            "bounds":[[0,10],[0.2,0.3],[0.04,0.06],[4.13*0.5,4.13*1.5],[0.07*0.5,0.07*1.5]],
            "dists":["unif","unif","unif","unif","unif"]}

S_sample=sobol.sample(problem,n,calc_second_order=False)

S_sampledf=pd.DataFrame(S_sample, columns=["homophily","local_ratio","noise_ratio","shock_alpha","shock_beta"])

# drop duplicates 
S_sampledf=S_sampledf.drop_duplicates()

# reindex
S_sampledf.index = pd.RangeIndex(start=1, stop=len(S_sampledf) + 1, step=1)
S_sampledf.index.name="RunID"

S_sampledf.to_csv("SaltelliSampleParams-n1024.csv")


