import sys
sys.path.append('../dgl_ptm')
import dgl_ptm
import os
import torch
import argparse
import collections
os.environ["DGLBACKEND"] = "pytorch"

# keeping None as default to make error obvious
parser = argparse.ArgumentParser(description='Sensitivity Analysis')
parser.add_argument('-r', '--run_id', default=None, type=int, help='Run ID from sample file (default: None)')
parser.add_argument('-s', '--seed', default=None, type=int, help='Seed (default: None)')
parser.add_argument('-ho', '--homophily', default=None, type=float, help='Homophily (default: None)') 
parser.add_argument('-l', '--local', default=None, type=float, help='Local attachment ratio (default: None)')
parser.add_argument('-n', '--noise', default=None, type=float, help='Noise ratio (default: None)')
parser.add_argument('-a', '--shock_a', default=None, type=float, help='Shock alpha (default: None)')
parser.add_argument('-b', '--shock_b', default=None, type=float, help='Shock beta (default: None)')

args =parser.parse_args()

model = dgl_ptm.PovertyTrapModel(model_identifier=f'SA_{args.run_id}_seed_{args.seed}',)

model.set_model_parameters(**{'number_agents': 10000 , 
    'seed':args.seed,
    'sigma_dist': {'type':'uniform','parameters':[0.05,1.94],'round':True,'decimals':1},
    'a_theta_dist': {'type':'uniform','parameters':[0.5,1],'round':False,'decimals':None},
    'sensitivity_dist':{'type':'uniform','parameters':[0.0,1],'round':False,'decimals':None},
    'capital_dist': {'type':'uniform','parameters':[0.1,10.],'round':False,'decimals':None}, 
    'alpha_dist': {'type':'normal','parameters':[1.08,0.074],'round':False,'decimals':None},
    'lambda_dist': {'type':'uniform','parameters':[0.05,0.94],'round':True,'decimals':1},
    'initial_graph_type': 'barabasi-albert',
    'initial_graph_args': {'seed': args.seed, 'new_node_edges':5},
    'device': 'cuda',
    'step_target':100,
    'steering_parameters':{'npath':'./agent_data.zarr',
                            'epath':'./edge_data', 
                            'ndata':['all_except',['a_table']],
                            'edata':None,
                            'mode':'w',
                            'capital_method':'past_shock',
                            'wealth_method':'weighted_transfer',
                            'income_method':'income_generation',
                            'tech_gamma': torch.tensor([0.3,0.35,0.45]),
                            'tech_cost': torch.tensor([0,0.15,0.65]),
                            'consume_method':'past_shock_bellman_consumption',
                            'nn_path': "/nn_data/both_PudgeSixLayer_1024/0723_110813/model_best.pth",
                            'adapt_m':torch.tensor([0,0.5,0.9]),
                            'adapt_cost':torch.tensor([0,0.25,0.45]),
                            'depreciation': 0.08,
                            'discount': 0.95,
                            'm_theta_dist': {'type':'beta','parameters':[args.shock_a,args.shock_b],'round':False,'decimals':None},
                            'del_method':'size',
                            'del_threshold':'balance',
                            'noise_ratio': args.noise,
                            'local_ratio': args.local,
                            'homophily_parameter': args.homophily,
                            'characteristic_distance':3.33, 
                            'truncation_weight':1.0e-10,
                            'step_type':'ptm'}})

print(model.config.steering_parameters)                        

model.initialize_model()
print (model.steering_parameters['modelTheta'] )
model.run()

