#This code generates data for two figures with four subplots of a)default, b) no social, 
# c) no adaptation, and d) null, i.e., no adaptation or social capital,
# model arrangements.
import zarr
import numpy as np
import pandas as pd
import os
from matplotlib.ticker import FixedLocator
from scipy.stats import wasserstein_distance
'''

#The root path:
output_path= "D:/UvA-RD/2026ReviewResponse/output/Disruption"

#The data folder paths:
arrangement_paths = ["disrupt_4_f3", "disrupt_5_f3"]
arrangement_labels = ["default_"]


completed_seeds=[15796,861,76821,54887,6266,
                82387,37195,87499,44132,60264,
                16024,41091,67222,64821,770,
                59736,62956,64926,67970,93017,
                83105,53708,85306,28694,71933]           

#The target data:
graph_test_data=False

#ts_target=[25,50,75]
#target_columns=False
ts_target=range(51)
target_columns=["i_a","wealth"]
full_initial_data=True
accumulations=True




def zarr_group_to_df(zarr_group, time_step=":", target_columns=False):
   

    df = pd.DataFrame()
    for array_name in zarr_group.array_keys():
        if target_columns is False or array_name in target_columns:
            zarr_array = zarr_group[array_name][:,time_step]
            df[array_name] = zarr_array.flatten()
            if array_name == "i_a" and accumulations!=False:
                zarr_array = zarr_group[array_name][:,0:time_step]
                df[f"{array_name}_accumulated"] = np.sum(zarr_array, axis=1)
            if array_name in ["theta","degree","wealth"] and full_initial_data!=False:
                zarr_array = zarr_group[array_name][:,0]
                df[f"{array_name}_initial"] = zarr_array.flatten()
    df.index.name = "AgentID"
    return df

# Create a dataframe from all available seeds for a model arrangement
# a target timestep specified above is used as a filter.

data_frames = {
    'disrupt_4_f3': pd.DataFrame(),
    'disrupt_5_f3': pd.DataFrame()
}

for arrangement_path in arrangement_paths:
    for _,dirnames,_ in os.walk(os.path.join(output_path,arrangement_path)):
        for folder_name in dirnames:
            if folder_name.startswith(arrangement_labels[0]):
                seed=int(folder_name.split('_')[-1])
                if seed in completed_seeds:
                    print(f"Processing {os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')}")
                    for ts in ts_target:
                        zarr_path = os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')
                        zarr_array = zarr.open(zarr_path, mode='r')
                        working_df = zarr_group_to_df(zarr_array, time_step=ts, target_columns=target_columns)
                        working_df['seed']=seed
                        working_df['time_step']=ts
                        working_df["AgentID"] = working_df.index
                        data_frames[arrangement_path] = pd.concat([data_frames[arrangement_path], working_df], ignore_index=True)

dfs = list(data_frames.values())
labels = list(data_frames.keys())
print(data_frames["disrupt_4_f3"].head())

disrupt_4_f3_i_a = pd.DataFrame()
arrangement_paths = ["disrupt_4_f3"]
for arrangement_path in arrangement_paths:
    for _,dirnames,_ in os.walk(os.path.join(output_path,arrangement_path)):
        for folder_name in dirnames:            
            if folder_name.startswith(arrangement_labels[0]):
                seed=int(folder_name.split('_')[-1])
                if seed in completed_seeds:
                    print(f"Processing {os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')}")
                    for ts in range(50):
                        zarr_path = os.path.join(output_path,arrangement_path,folder_name,'agent_data.zarr')
                        zarr_array = zarr.open(zarr_path, mode='r')
                        working_df = zarr_group_to_df(zarr_array, time_step=ts, target_columns="i_a")
                        working_df['seed']=seed
                        working_df['time_step']=ts
                        working_df['AgentID'] = working_df.index
                        disrupt_4_f3_i_a = pd.concat([disrupt_4_f3_i_a, working_df], ignore_index=True)

i_a_dfs = list(data_frames.values())
i_a_labels = list(data_frames.keys())
print(disrupt_4_f3_i_a.head())
# Switching plot 


## upgrades and reductions
disrupt_4_f3_i_a["i_a_previous"] = disrupt_4_f3_i_a.groupby(["seed","AgentID"])["i_a"].shift(1)
disrupt_4_f3_i_a["upgrades"] = (disrupt_4_f3_i_a["i_a_previous"] < disrupt_4_f3_i_a["i_a"]).astype(int)
disrupt_4_f3_i_a["reductions"] = (disrupt_4_f3_i_a["i_a_previous"] > disrupt_4_f3_i_a["i_a"]).astype(int)
switch_df=pd.DataFrame({'time_step':range(50)})
switch_df['upgrades'] = disrupt_4_f3_i_a[disrupt_4_f3_i_a["time_step"]>0].groupby("time_step")["upgrades"].sum()/disrupt_4_f3_i_a[disrupt_4_f3_i_a["time_step"]>0].groupby("time_step").size().tolist()
switch_df["reductions"] = disrupt_4_f3_i_a[disrupt_4_f3_i_a["time_step"]>0].groupby("time_step")["reductions"].sum()/disrupt_4_f3_i_a[disrupt_4_f3_i_a["time_step"]>0].groupby("time_step").size().tolist()
switch_df.to_csv("AdditionalExperiments/Data/disrupt_4_f3_switches.csv", index=False)


def save_boxplot_data(total_investment, delta_k, filename, bin_size=5):
    max_investment = int(np.max(total_investment))
    bins = np.arange(0, max_investment + bin_size, bin_size)
    
    all_data = []
    for i in range(len(bins) - 1):
        bin_mask = (total_investment >= bins[i]) & (total_investment < bins[i + 1])
        bin_data = delta_k[bin_mask]
        if len(bin_data) > 0:
            min_val = np.min(bin_data)
            q1 = np.percentile(bin_data, 25)
            median = np.median(bin_data)
            q3 = np.percentile(bin_data, 75)
            max_val = np.max(bin_data)
            all_data.append({
                "bin_start": bins[i],
                "bin_end": bins[i + 1],
                "min": min_val,
                "q1": q1,
                "median": median,
                "q3": q3,
                "max": max_val
            })
    boxplot_data = pd.DataFrame(all_data)
    boxplot_data.to_csv(filename, index=False)
    

# boxplot data for capital at final time step vs. total investment in adaptation


total_investment = data_frames['disrupt_4_f3']["i_a_accumulated"]
delta_k = data_frames['disrupt_4_f3']["wealth"]-data_frames['disrupt_4_f3']["wealth_initial"]
save_boxplot_data(total_investment, delta_k, 'AdditionalExperiments/Data/boxplot_bin_size_5.csv')




def save_boxplot_comparison(dfs, labels, filename, bin_size=4):
    all_data = []
    for df, label in zip(dfs, labels):
        df=df[df["time_step"]==df["time_step"].max()]
        df.loc[:, "delta_k"] = df["wealth"] - df["wealth_initial"]
        i_a = df["i_a_accumulated"]
        delta_k = df["delta_k"]
        
        bins = np.arange(min(i_a), max(i_a) + bin_size, bin_size)
        for i in range(len(bins) - 1):
            bin_mask = (i_a >= bins[i]) & (i_a < bins[i + 1])
            bin_data = delta_k[bin_mask]
            if len(bin_data) > 0:
                min_val = np.min(bin_data)
                q1 = np.percentile(bin_data, 25)
                median = np.median(bin_data)
                q3 = np.percentile(bin_data, 75)
                max_val = np.max(bin_data)
                all_data.append({
                    "bin_start": bins[i],
                    "bin_end": bins[i + 1],
                    "min": min_val,
                    "q1": q1,
                    "median": median,
                    "q3": q3,
                    "max": max_val,
                    "label": label
                })
    
    comparison_data = pd.DataFrame(all_data)
    comparison_data.to_csv(filename, index=False)

save_boxplot_comparison(dfs,['Disruption 0.4', 'Disruption 0.5'], 'AdditionalExperiments/Data/disruption_comparison_boxplot_bin_size_4.csv')

def calculate_histogram_data(df, column, bins):
    counts, bin_edges = np.histogram(df[column], bins=bins, density=True)
    percent_counts = counts * 100 
    return bin_edges[:-1], percent_counts

def compute_wasserstein_distance(dfs, labels, column="wealth"):
    distances = []
    for df, label in zip(dfs, labels):
        for seed in df['seed'].unique():
            seed_df = df[df['seed'] == seed]
            time_steps = sorted(seed_df['time_step'].unique())
            for t0, t1 in zip(time_steps[:-1], time_steps[1:]):
                bins = np.arange(0, seed_df[column].max() + 1, 1)
                bin_edges, counts_t0 = calculate_histogram_data(seed_df[seed_df['time_step'] == t0], column, bins)
                bin_edges, counts_t1 = calculate_histogram_data(seed_df[seed_df['time_step'] == t1], column, bins)
                sd=seed_df[seed_df['time_step'] == t1][column].std()
                distance = wasserstein_distance(counts_t0, counts_t1)
                distances.append({"Disruption": label, "seed": seed, "t_0": t0, "t_1": t1, "Wasserstein_distance": distance, "k_t_1_sdev":sd})
    return pd.DataFrame(distances)

# Compute and save Wasserstein distances
wasserstein_distances = compute_wasserstein_distance(dfs, labels)
wasserstein_distances.to_csv(f"AdditionalExperiments/Data/Wasserstein_distances.csv", index=False)


def save_histogram_data(dfs, labels, hist_ts, filename, column="wealth"):
    all_data = []
    for df, label in zip(dfs, labels):
        print(df["time_step"].unique())
        print(df["time_step"].dtype)
        print(hist_ts, type(hist_ts))
        print(df[df["time_step"] == hist_ts].head())
        df=df[df["time_step"]==hist_ts]

        print(df.head())
        bins = np.arange(0, df[column].max() + 1, 1)
        bin_edges, percent_counts = calculate_histogram_data(df, column, bins)
        for bin_edge, percent_count in zip(bin_edges, percent_counts):
            all_data.append({"Disruption": label, "bin": bin_edge, "frequency": percent_count})
    hist_data = pd.DataFrame(all_data)
    hist_data.to_csv(filename, index=False)

# Save histogram data for all scenarios in a single CSV file
hist_ts = dfs[0]['time_step'].max()
print(hist_ts)
save_histogram_data(dfs, labels, hist_ts ,f"AdditionalExperiments/Data/wealth_histogram_data_t{hist_ts}.csv")
'''
# Plot Time!
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.patheffects as path_effects
from matplotlib.lines import Line2D
from matplotlib.text import Text
from matplotlib.legend_handler import HandlerBase
from matplotlib.collections import LineCollection,PatchCollection



import plotly.subplots
import plotly.graph_objs
import plotly.io as pio

plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.rm'] = 'Times New Roman'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['mathtext.bf'] = 'Times New Roman:bold'
plt.rcParams['font.size'] = 12

'''

class HandlerHatchedRectangle(HandlerBase):
    """Custom legend handler that draws a rectangle filled with diagonal lines."""
    
    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        # Create the base rectangle
        rect = patches.Rectangle((xdescent, ydescent), width, height,
                                 facecolor='none', edgecolor=orig_handle.get_edgecolor(),
                                 linewidth=orig_handle.get_linewidth(), transform=trans)

        # Manually create diagonal lines inside the rectangle
        diagonal_lines = []
        num_lines = 5  # Adjust density of diagonal lines
        for i in np.linspace(0, width, num_lines):
            diagonal_lines.append([(xdescent + i, ydescent), (xdescent + i - height, ydescent + height)])

        # Create line collection
        lines = [Line2D([p1[0], p2[0]], [p1[1], p2[1]], color=orig_handle.get_edgecolor(), 
                         linewidth=orig_handle.get_linewidth(), transform=trans) 
                 for p1, p2 in diagonal_lines]

        return [rect] + lines  # Return both the rectangle and the diagonal lines
'''

switches_df = pd.read_csv('AdditionalExperiments/Data/disrupt_4_f3_switches.csv')

# Create a figure and subplots
fig, axs = plt.subplots( 1,2, figsize=(10, 4))


#Bar chart of upgrades and reductions by time step
for timestep in np.arange(2, switches_df["time_step"].max() + 1, 3):
        axs[0].axvline(timestep, color='#3f7bc1', alpha=1, linestyle='-', linewidth=0.2)
        axs[0].axvline(timestep, color='#3f7bc1', alpha=0.5, linestyle='-', linewidth=0.5)
        axs[0].axvline(timestep, color='#3f7bc1', alpha=0.2, linestyle='-', linewidth=1)



#    axs[0].bar(timestep, 46, width=1, color='none', edgecolor='#3f7bc1', alpha=1, linewidth=0.2)
#    axs[0].bar(timestep, -46, width=1, color='none', edgecolor='#3f7bc1', alpha=1,linewidth=0.2)
#    for y in np.arange(-46, 46, 1): 
#        line = LineCollection([[(timestep-.5, y), (timestep + .5, y + 1)]], colors='#3f7bc1', linewidths=0.2, alpha=1)
#        axs[0].add_collection(line)
axs[0].bar(switches_df["time_step"], switches_df["upgrades"]*100, color='#678d58', edgecolor=(0,0,0,0.5), width=1,label='Upgrades',zorder=10)#,align='edge')
axs[0].bar(switches_df["time_step"], -switches_df["reductions"]*100, color='#b53530', edgecolor=(0,0,0,0.5), width=1,label='Reductions')#,align='edge')
axs[0].set_xlabel("Timestep")
axs[0].set_ylabel("Percentage of Agents Switching")
axs[0].set_title("(a)")
#legend_proxy = patches.Rectangle((0, 0), 1, 1, facecolor='none', edgecolor='#3f7bc1', linewidth=0.2,label='$\\Theta$ = 0.7 Disruption')

#legend_handles = [Line2D([0], [0], color=color, marker='s', markersize=10, linestyle='None', markeredgecolor=(0,0,0,0.5), label=label) for color, label in zip(['#678d58','#b53530'], ['Upgrades','Reductions'])]
legend_handles = [
    Line2D([0], [0], color='#678d58', marker='s', markersize=10, linestyle='None', markeredgecolor=(0, 0, 0, 0.5), label='Upgrades'),
    Line2D([0], [0], color='#b53530', marker='s', markersize=10, linestyle='None', markeredgecolor=(0, 0, 0, 0.5), label='Reductions'),
    Line2D([0], [0], color='#3f7bc1', alpha=1, marker='_', markersize=10, linestyle='None', label='$\\Theta$ = 0.4 Disruption')]
    #legend_proxy]
legend=axs[0].legend(handles=legend_handles,loc='upper left', frameon=False,labelspacing=0.2, handletextpad= 0,bbox_to_anchor=(-0.04, 1.02))
#axs[0].legend(handles=legend_handles, loc='upper left', frameon=False,
#          handler_map={patches.Rectangle: HandlerHatchedRectangle()})
for text in legend.get_texts():
    text.set_path_effects([
        path_effects.Stroke(linewidth=3, foreground='white'),
        path_effects.Normal() 
    ])
axs[0].set_yticks([-100,-80,-60,-40,-20, 0, 20, 40, 60, 80, 100])
axs[0].set_yticklabels(["100%","80%","60%","40%","20%", '0%', "20%", "40%", "60%", "80%", "100%"])
axs[0].set_xlim(0, 50)
axs[0].set_ylim(-100, 100)

# Adjust layout and save the figure
#plt.savefig('AdditionalExperiments/Figures/SevereSwitches.pdf')
#plt.show()

class HandlerRightLabel(HandlerBase):
    def __init__(self, label_map):
        super().__init__()
        self.label_map = label_map  # Store label mapping

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        label = self.label_map.get(orig_handle, '')  # Retrieve the correct label

        # Create text label first
        text = Text(width - 103, height / 2 - 1, label, verticalalignment='center', 
                    horizontalalignment='left', transform=trans)

        # Create marker manually
        handle = Line2D([width], [height / 2], color=orig_handle.get_color(), 
                        marker=orig_handle.get_marker(), markersize=orig_handle.get_markersize(), 
                        linestyle='None', markeredgecolor=orig_handle.get_markeredgecolor(), 
                        transform=trans)

        return [text, handle] 


comparison_data = pd.read_csv('AdditionalExperiments/Data/disruption_comparison_boxplot_bin_size_4.csv')
bin_size = 4
# Group data by bins and labels
bins = comparison_data['bin_start'].unique()
bins.sort()
labels = comparison_data['label'].unique()
binned_data = {label: [] for label in labels}
for label in labels:
    for bin_start in bins:
        bin_data = comparison_data[(comparison_data['bin_start'] == bin_start) & (comparison_data['label'] == label)]
        binned_data[label].append(bin_data[['min', 'q1', 'median', 'q3', 'max']].values.flatten())
# Create boxplot
positions = {label: bins[:] + bin_size / 2 +(i - (len(labels) - 1) / 2) * (bin_size / (len(labels) + 1)) for i, label in enumerate(labels)}
#colors = ['midnightblue', 'steelblue', 'lightsteelblue']
colors = ['#254a74', '#3f7bc1']
legend_labels = ["$\\Theta$ = 0.4 Disruption", "$\\Theta$ = 0.5 Disruption"]


for i, label in enumerate(labels):
    for j, bin_data in enumerate(binned_data[label][:len(positions[label])]):
        axs[1].bxp([{
            'med': bin_data[2],
            'q1': bin_data[1],
            'q3': bin_data[3],
            'whislo': bin_data[0],
            'whishi': bin_data[4]
       # }], positions=[positions[label][j]], widths=bin_size / 6, showfliers=False, patch_artist=True, boxprops=dict(facecolor=colors[i],edgecolor=(0,0,0,0.5)), medianprops=dict(color="black", alpha=0.5),whiskerprops=dict(color="black", alpha=0.5))
        }], positions=[positions[label][j]], widths=bin_size / 4, showfliers=False, patch_artist=True, boxprops=dict(facecolor=colors[i],edgecolor=(0,0,0,0.5)), medianprops=dict(color="black", alpha=0.5),whiskerprops=dict(color="none"))

axs[1].set_xlabel("Total $i_a$")
axs[1].set_ylabel("Δ $k$")
axs[1].set_title("(b)")
#axs[1].set_ylim(-15, 150)
axs[1].set_ylim(-5, 25)
axs[1].set_xticks(bins + bin_size / 2)
axs[1].set_xticklabels([f'{int(bins[i])}-{int(bins[i]) + bin_size}' for i in range(len(bins))])
legend_handles = [Line2D([0], [0], color=color, marker='s', markersize=10, linestyle='None', markeredgecolor=(0,0,0,0.5), label="") for color, label in zip(colors, labels)]
axs[1].legend(handles=legend_handles, loc='upper right', frameon=False, handler_map={Line2D: HandlerRightLabel({handle: label for handle, label in zip(legend_handles, legend_labels)})}, handlelength= 0,labelspacing=0.2,bbox_to_anchor=(1.03, 1.02))#, handletextpad=0, labelspacing=0)
plt.tight_layout()
plt.savefig('AdditionalExperiments/Figures/switches_delta_k_vs_i_a_comparison.pdf')
plt.show()


def plot_WD_data(df, label, title, ax, color):
    point_data = df[df["Disruption"] == label]
    ax.scatter(point_data["t_1"], point_data["WD_mean"], color=color,marker='.')
    ax.errorbar(point_data["t_1"], point_data["WD_mean"], yerr=point_data["WD_sd"], fmt='none', ecolor='lightgrey', alpha=0.5, capsize=4)
    ax.plot(point_data["t_1"], (point_data["k_sd"]*0.01), color='black', alpha=0.5, linestyle=':',label='1% of $k_t$ S.D.')
    ax.set_title(title,fontsize=plt.rcParams['font.size'])
    ax.set_xlim(0, 50)
    ax.set_xlabel('Time Step')
    ax.set_ylabel('Wasserstein Distance')
    ax.legend(frameon=False)

data = pd.read_csv("AdditionalExperiments/Data/Wasserstein_distances.csv")

data = data.groupby(['Disruption', 't_1']).agg(WD_mean=('Wasserstein_distance', 'mean'), WD_sd=('Wasserstein_distance', 'std'), k_sd=('k_t_1_sdev','min')).reset_index()
print(data)
fig_wdk, axs_wdk = plt.subplots( 1,2, figsize=(10, 4))

# Plot wasserstien distance data for each scenario
plot_WD_data(data, "disrupt_4_f3", "(a) $\\Theta$ = 0.4 Disruption", axs_wdk[0], '#254a74')
plot_WD_data(data, "disrupt_5_f3", "(b) $\\Theta$ = 0.5 Disruption", axs_wdk[1], '#3f7bc1')


# Set matching y-axis limits for all subplots
max_ylim = max(ax.get_ylim()[1] for ax in axs_wdk.flat)
for ax in axs_wdk.flat:
    ax.set_ylim(0, max_ylim)
    #y_ticks = ax.get_yticks()
    #ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    #ax.set_yticklabels([f'{ y:.0f}%' for y in y_ticks])

plt.tight_layout()
plt.savefig("AdditionalExperiments/Figures/2Panel_WD_k.pdf")
plt.show()

data = pd.read_csv("AdditionalExperiments/Data/wealth_histogram_data_t50.csv")
print(data.head())

def plot_histogram_data(df, label, title, ax, color, outline='white'):
    hist_data = df[df["Disruption"] == label]
    ax.bar(hist_data["bin"], hist_data["frequency"], color=color, width=1.0,edgecolor=outline, linewidth=0.05)
    ax.set_title(title,fontsize=plt.rcParams['font.size'])
    ax.set_xlim(0, 50)
    ax.set_xlabel('Wealth, $k_t$')
    ax.set_ylabel('Percent Frequency')
    print(hist_data['frequency'][hist_data['bin'] >= 50])
    uncaptured = (hist_data['frequency'][hist_data['bin'] >= 50]).sum()
    ax.text(0.95, 0.95, f'$k_t$ > 50: {uncaptured:.2f}%', transform=ax.transAxes, fontsize=8, verticalalignment='top', horizontalalignment='right')


fig_hk, axs_hk = plt.subplots(1, 2, figsize=(10, 4))

# Plot histogram data for each scenario
plot_histogram_data(data, "disrupt_4_f3", "(a) $\\Theta$ = 0.4 Disruption", axs_hk[ 0], '#254a74','#b2cae6')
plot_histogram_data(data, "disrupt_5_f3", "(b) $\\Theta$ = 0.5 Disruption", axs_hk[1], '#3f7bc1','#b2cae6')

# Set matching y-axis limits for all subplots
max_ylim = max(ax.get_ylim()[1] for ax in axs_hk.flat)
for ax in axs_hk.flat:
    ax.set_ylim(0, max_ylim)
    y_ticks = ax.get_yticks()
    ax.yaxis.set_major_locator(FixedLocator(y_ticks))
    ax.set_yticklabels([f'{ y:.0f}%' for y in y_ticks])

plt.tight_layout()
plt.savefig("AdditionalExperiments/Figures/2Panel_hist_k.pdf")

