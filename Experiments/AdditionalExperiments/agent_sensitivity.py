import zarr
from SALib.analyze import pawn
import numpy as np
import pandas as pd
import os
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['font.size'] = 12

ts_target=[50]

output_path= "D:/UvA-RD/Output"
#The data folder paths:
arrangement_paths = ["default", "no_social"]
arrangement_labels = ["default_", "no_social_"]


seeds=[861,76821,54887,6266,87499,
       60264,67222,770,59736,67970,
       83105,53708,71933,85306,93017]        

full_initial_data=True
accumulations=True
target_columns=["wealth", "i_a", "wealth_consumption", "theta", "degree", "weighted_degree"]



def zarr_group_to_df(zarr_group, time_step=":", target_columns=False):
    """
    Convert a zarr group to a pandas dataframe.
    """

    df = pd.DataFrame()
    for array_name in zarr_group.array_keys():
        if target_columns is False or array_name in target_columns:
            zarr_array = zarr_group[array_name][:,time_step]
            df[array_name] = zarr_array.flatten()
            if array_name == "i_a" and accumulations!=False:
                zarr_array = zarr_group[array_name][:,0:time_step]
                df[f"{array_name}_accumulated"] = np.sum(zarr_array, axis=1)
                print(f'sum:{np.sum(np.sum(zarr_array, axis=1))}')
            if array_name in ["theta","degree","wealth"] and full_initial_data!=False:
                zarr_array = zarr_group[array_name][:,0]
                df[f"{array_name}_initial"] = zarr_array.flatten()
    df.index.name = "AgentID"
    return df

def get_bounds(column_name, df):
    min_val = df[column_name].min()
    max_val = df[column_name].max()
    return [min_val, max_val]



data_frames = {
    'default': pd.DataFrame(),
    'no_social': pd.DataFrame(),
}
for arrangement_path in arrangement_paths:
    for _,dirnames,_ in os.walk(os.path.join(output_path,arrangement_path)):
        for folder_name in dirnames:
            if folder_name.startswith(f"{arrangement_path}_"):
                seed=int(folder_name.split('_')[-1])
                if seed in seeds:
                    print(f"Processing {os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')}")
                    for ts in ts_target:
                        zarr_path = os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')
                        zarr_array = zarr.open(zarr_path, mode='r')
                        working_df = zarr_group_to_df(zarr_array, time_step=ts, target_columns=target_columns)
                        working_df['seed']=seed
                        working_df['time_step']=ts
                        if full_initial_data==True:
                            zarr_path = os.path.join(output_path,arrangement_path,folder_name, 'agent_data_initial.zarr')
                            zarr_array = zarr.open(zarr_path, mode='r')
                            agent_df = zarr_group_to_df(zarr_array,time_step=0)
                            working_df=pd.merge(working_df, agent_df, on="AgentID")
                        data_frames[arrangement_path] = pd.concat([data_frames[arrangement_path], working_df], ignore_index=True)

default_df = data_frames['default']
default_df.to_csv("AdditionalExperiments/Data/default_df.csv", index=False)
no_social_df = data_frames['no_social']
no_social_df.to_csv("AdditionalExperiments/Data/no_social_df.csv", index=False)
print(default_df.head())
dfs = [default_df, no_social_df]
labels = ["default_arrangement", "no_social_arrangement"]

default_ks = pd.DataFrame()

for seed in seeds:
    data=default_df[default_df['seed']==seed]
    default_problem = {"num_vars": 9, 
           "names": ["wealth_initial","theta_initial",
                    "degree_initial","sensitivity","lambda", 
                    "sigma",  "alpha", 
                    "i_a_accumulated",
                    "weighted_degree"], 
            "bounds": [[0.1,10], [0.1,1],
                       get_bounds("degree_initial", data), [0,1],[0.5,0.95],
                       get_bounds("sigma", data),get_bounds("alpha", data), 
                       get_bounds("i_a_accumulated", data), 
                       get_bounds("weighted_degree", data)]}

    default_pawn=pawn.analyze(default_problem,data[[ "wealth_initial","theta_initial","degree_initial","sensitivity","lambda", "sigma",  "alpha", "i_a_accumulated","weighted_degree"]].values, data["wealth"].values, S=20,print_to_console=True)
    default_ks = pd.concat([default_ks, pd.DataFrame([{"seed": seed, **dict(zip(default_pawn['names'], default_pawn['median']))}])], ignore_index=True)


no_social_ks = pd.DataFrame()

for seed in seeds:
    data=no_social_df[no_social_df['seed']==seed]
    no_social_problem = {"num_vars": 9, 
           "names": ["wealth_initial","theta_initial",
                    "degree_initial","sensitivity","lambda", 
                    "sigma",  "alpha", 
                    "i_a_accumulated",
                    "weighted_degree"], 
            "bounds": [[0.1,10], [0.1,1],
                       get_bounds("degree_initial", data), [0,1],[0.5,0.95],
                       get_bounds("sigma", data),get_bounds("alpha", data), 
                       get_bounds("i_a_accumulated", data), 
                       get_bounds("weighted_degree", data)]}

    no_social_pawn=pawn.analyze(no_social_problem,data[[ "wealth_initial","theta_initial","degree_initial","sensitivity","lambda", "sigma",  "alpha", "i_a_accumulated","weighted_degree"]].values, data["wealth"].values, S=20,print_to_console=True)
    no_social_ks = pd.concat([no_social_ks, pd.DataFrame([{"seed": seed, **dict(zip(no_social_pawn['names'], no_social_pawn['median']))}])], ignore_index=True)
colors=['#342d49','#8b7d4f','#3f7bc1','#d15853']
edgecolors=['#70619e','#b6a87c','#9fbde0','#e49d9a']
# Default KS medians boxplot

fig, ax = plt.subplots(figsize=(10, 5))
ax.boxplot([default_ks["wealth_initial"], default_ks["theta_initial"], 
            default_ks["degree_initial"], default_ks["sensitivity"], 
            default_ks["lambda"], default_ks["sigma"], default_ks["alpha"], 
            default_ks["i_a_accumulated"], default_ks["weighted_degree"]], 
            tick_labels=['Initial\nWealth,\n$k_i$','Initial\nShock\nPerception,\n$\\theta_i$',
                    'Initial\nDegree', 'Sensitivity', 'Savings\nPropensity,\n$\\lambda$',
                    'Risk\nAversion,\n$\\sigma$', 'Human\nCapital,\n$\\alpha$',
                    'Gross\nAdaptation\nInvestment,\n$i_{a,total}$', 'Weighted\nDegree'],
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='#342d49', markeredgecolor='white', markersize=6, markeredgewidth=0.3),
                    medianprops=dict(color="#342d49"),
                    boxprops=dict( color="#342d49"),
                    whiskerprops=dict(color="#342d49"),
                    capprops=dict(color="#342d49"))
ax.set_ylabel('Median KS Across 20 Slides')
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.savefig("AdditionalExperiments/Figures/default_sensitivity.pdf")
plt.show()

# No Social KS medians boxplot

fig2, ax2 = plt.subplots(figsize=(10, 5))
ax2.boxplot([no_social_ks["wealth_initial"], no_social_ks["theta_initial"], 
            no_social_ks["degree_initial"], no_social_ks["sensitivity"], 
            no_social_ks["lambda"], no_social_ks["sigma"], no_social_ks["alpha"], 
            no_social_ks["i_a_accumulated"], no_social_ks["weighted_degree"]], 
            tick_labels=['Initial\nWealth,\n$k_i$','Initial\nShock\nPerception,\n$\\theta_i$',
                    'Initial\nDegree', 'Sensitivity', 'Savings\nPropensity,\n$\\lambda$',
                    'Risk\nAversion,\n$\\sigma$', 'Human\nCapital,\n$\\alpha$',
                    'Gross\nAdaptation\nInvestment,\n$i_{a,total}$', 'Weighted\nDegree'],
                    showmeans=True, meanprops=dict(marker='D', markerfacecolor='#8b7d4f', markeredgecolor='white', markersize=6, markeredgewidth=0.3),
                    medianprops=dict(color="#8b7d4f"),
                    boxprops=dict( color="#8b7d4f"),
                    whiskerprops=dict(color="#8b7d4f"),
                    capprops=dict(color="#8b7d4f"))
ax2.set_ylabel('Median KS Across 20 Slides')
plt.subplots_adjust(bottom=0.25)
plt.tight_layout()
plt.savefig("AdditionalExperiments/Figures/no_social_sensitivity.pdf")
plt.show()
