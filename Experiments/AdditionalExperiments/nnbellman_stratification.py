
import sys
sys.path.append('../dgl_ptm')
import dgl_ptm
from dgl_ptm.util.utils import load_consumption_model
import dgl_ptm.util.nn_arch.nn_arch as nn_arch
from dgl_ptm.util.utils import scale_input
import pandas as pd
import numpy as np
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import matplotlib.pyplot as plt
plt.rcParams['font.family'] = 'Times New Roman'
plt.rcParams['mathtext.fontset'] = 'custom'
plt.rcParams['mathtext.it'] = 'Times New Roman:italic'
plt.rcParams['font.size'] = 12


device = torch.device("cuda")
adapt_m=torch.tensor([0,0.5,0.9])
adapt_cost=torch.tensor([0,0.2,0.5])
i_a_mapping = {'N': 0.0, 'L': 0.2, 'H': 0.5}

# inport an input dataframe and iteratively solved solutions
df = pd.read_csv("D:/UvA-RD/2026ReviewResponse/NeuralNetwork/DatasetOct/AgentData-Updated30Oct2024.csv")

print(df.shape)
               
iterative_solutions = pd.read_csv("D:/UvA-RD/2026ReviewResponse/NeuralNetwork/DatasetOct/ResultsFinal-Updated30Oct2024clean.csv")

print(iterative_solutions.shape)


df = df.merge(iterative_solutions, on="AgentID", how="inner")
df['i_a'] = df['Equation'].map(i_a_mapping)

print(df.shape[0])
print(df.columns)
a_table= torch.stack([adapt_m,adapt_cost]).repeat(int(df.shape[0]),1,1).to(device)
model_path= "/nn_data/both_PudgeSixLayer_2048/1106_002520/model_best.pth"

#print("entered load_consumption_model")

estimator,cons_scale, i_a_scale,input_scale = load_consumption_model(model_path,device)  

estimator.to(device)
estimator.eval()

input = torch.tensor(df[['Alpha', 'k', 'Sigma', 'Theta']].values, dtype=torch.float32, device=device)

# Scale inputs as specified in nn config
if {"alpha","Alpha"} & input_scale.keys():
    input[:,0]=scale_input(input[:,0], input_scale.get(list({"alpha","Alpha"} & input_scale.keys())[0]),"alpha",False)
if {"k","K"} & input_scale.keys():
    input[:,1]=scale_input(input[:,1], input_scale.get(list({"k","K"} & input_scale.keys())[0]),"k",False)
if {"sigma","Sigma"} & input_scale.keys():
    input[:,2]=scale_input(input[:,2], input_scale.get(list({"sigma","Sigma"} & input_scale.keys())[0]),"sigma",False)
if {"theta","Theta"} & input_scale.keys():
    input[:,3]=scale_input(input[:,3], input_scale.get(list({"theta","Theta"} & input_scale.keys())[0]), "theta",False)

# Forward pass to get predictions
with torch.no_grad():

    pred=estimator(input)

#model_graph.ndata['m'],model_graph.ndata['i_a'] are initialized as zeros
# print("Cleaning output and checking for violations")



m,i_a=a_table[torch.arange(a_table.size(0)),:,torch.argmin(torch.abs(pred[:, 0].unsqueeze(1)*i_a_scale - a_table[:,1,:]), dim=1)].unbind(dim=1)
    

print(f"Setting {torch.sum((pred[:,1]*cons_scale)<0)} negative of {df.shape[0]} consumption predictions to zero, {torch.sum((pred[:,1]*cons_scale)<-0.1)} was/were less than -0.1.")
consumption=(pred[:,1]*cons_scale).clamp_(min=0)

df['predicted_i_a']=i_a.cpu().numpy()
df['predicted_consumption']=consumption.cpu().numpy()

print(df.head())
print(df['Alpha'].describe())
print(df['k'].describe())
print(df['Sigma'].describe())
print(df['Theta'].describe())

df['k_decile'] = pd.qcut(df['k'], 10, labels=False)
df['ae_consumption'] = (df['Consumption'] - df['predicted_consumption']).abs()
df['ae_i_a'] = (df['i_a'] - df['predicted_i_a']).abs()
df['wrong_i_a_weighted'] = np.where(df['ae_i_a'] > 0.4, 2, np.where(df['ae_i_a'] > 0.1, 1, 0))
df['wrong_i_a'] = np.where(df['ae_i_a'] > 0.4, 1, np.where(df['ae_i_a'] > 0.1, 1, 0))
consumption_error = [df[df['k_decile'] == i]['ae_consumption'].values for i in range(10)]
i_a_error = [df[df['k_decile'] == i]['ae_i_a'].values for i in range(10)]

print(f"Total wrong i_a predictions: {df['wrong_i_a'].sum()}")
print(f"Total wrong i_a predictions weighted by levels: {df['wrong_i_a_weighted'].sum()}")

fig1, ax1 = plt.subplots(figsize=(8, 5))
_, k_bins = pd.qcut(df['k'], 10, retbins=True, duplicates='drop')
for i in range(len(k_bins)-1):
    left = k_bins[i]
    right = k_bins[i+1]
    if right > 0.1 and left < 10:
        xpos = i + 1
        ax1.axvspan(xpos-0.5, xpos+0.5, color='gray', alpha=0.15)
box = ax1.boxplot(consumption_error, patch_artist=True, medianprops=dict(color="black", alpha=0.5),
                  showfliers=False,boxprops=dict(facecolor="grey", color="black", alpha=0.5),
                  showmeans=True,
                  meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red'))

ax1.set_xlabel("Wealth Decile")
ax1.set_ylabel("Absolute Error in Consumption")
ax1.set_title("Consumption Error by Wealth")
ax1.set_xticklabels([f"{i+1}" for i in range(10)])
plt.tight_layout()
plt.savefig("AdditionalExperiments/Figures/Consumption_Err_k.pdf")
plt.show()

fig2, ax2 = plt.subplots(figsize=(8, 5))
_, k_bins = pd.qcut(df['k'], 10, retbins=True, duplicates='drop')
for i in range(len(k_bins)-1):
    left = k_bins[i]
    right = k_bins[i+1]
    if right > 0.1 and left < 10:
        xpos = i + 1
        ax2.axvspan(xpos-0.5, xpos+0.5, color='gray', alpha=0.15)
box = ax2.boxplot(i_a_error, patch_artist=True, medianprops=dict(color="black", alpha=0.5),
                  showfliers=False,boxprops=dict(facecolor="grey", color="black", alpha=0.5),
                  showmeans=True,
                  meanprops=dict(marker='D', markeredgecolor='black', markerfacecolor='red'))

ax2.set_ylim(-0.001, 0.005)
ax2.set_xlabel("Wealth Decile")
ax2.set_ylabel("Absolute Error in $i_a$")
ax2.set_title("$i_a$ Error by Wealth")
ax2.set_xticklabels([f"{i+1}" for i in range(10)])
plt.tight_layout()
plt.savefig("AdditionalExperiments/Figures/i_a_Err_k.pdf")
plt.show()

fig3, ax3 = plt.subplots(figsize=(8, 5))
deciles = range(10)
correct = []
wrong_1 = []
wrong_2 = []
total = []

for i in deciles:
    group = df[df['k_decile'] == i]['wrong_i_a']
    correct.append((group == 0).sum())
    wrong_1.append((group == 1).sum())
    wrong_2.append((group == 2).sum())
    total.append(len(group))


bar1 = ax3.bar(deciles, correct, label='Correct', color='grey')
bar2 = ax3.bar(deciles, wrong_1, bottom=correct, label='Miscalculated by 1 Level', color='orange')
bar3 = ax3.bar(deciles, wrong_2, bottom=np.array(correct)+np.array(wrong_1), label='Miscalculated by 2 Levels', color='red')

for i, (c, t) in enumerate(zip(correct, total)):
    if t > 0:
        percent = 100 * c / t
        ax3.text(i, c/2, f"{percent:.1f}%", ha='center', va='center', color='white', fontsize=10, fontweight='bold')

ax3.set_xlabel("Wealth Decile")
ax3.set_ylabel("Count")
ax3.set_title("$i_a$ Misclassification by Wealth Decile")
ax3.set_xticks(deciles)
ax3.set_xticklabels([f"{i+1}" for i in deciles])
ax3.legend()
plt.tight_layout()
plt.savefig("AdditionalExperiments/Figures/i_a_Wrong_k_bar.pdf")
plt.show()