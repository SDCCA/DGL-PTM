
import torch
import numpy as np

def capital_update(model_graph, params=None, model_data=None, timestep=None, method='default'):
    # Calculate income generated   
    if method == 'default':
        _agent_capital_update(model_graph, params, model_data, timestep)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'default'")

def _agent_capital_update(model_graph,model_params,model_data,timestep):
    #formula for k_t+1 is applied at the beginning of each time step 
    # k_t+1 becomes the new k_t
    
    k,c,i_a,m,α = model_graph.ndata['wealth'],model_graph.ndata['wealth_consumption'],model_graph.ndata['i_a'],model_graph.ndata['m'],model_graph.ndata['alpha']
    global_θ =model_data['modelTheta'][timestep-1]
    𝛿=model_params['depreciation']
    model_graph.ndata['wealth'] = (global_θ + m * (1-global_θ)) * (model_graph.ndata['income'] - c - i_a + (1-𝛿) * k)
