
import torch
import numpy as np

def capital_update(model_graph, model_params=None, timestep=None, method='default'):
    # formula for k_t+1 is applied at the beginning of each time step 
    # the result for k_t+1 becomes the current k_t
    if method == 'present_shock':
        _agent_capital_update(model_graph, model_params, timestep)
    elif method == 'past_shock':
        _agent_capital_past_shock_update(model_graph, model_params, timestep)
    else:
        raise NotImplementedError("Incorrect capital update method received.")


def _agent_capital_update(model_graph,model_params,timestep):
    # Applies shock from previous time step to entire stock of new capital
    k,c,i_a,m = model_graph.ndata['wealth'],model_graph.ndata['wealth_consumption'],model_graph.ndata['i_a'],model_graph.ndata['m']
    global_θ = model_params['global_theta'][timestep-1]
    𝛿=model_params['depreciation']
    model_graph.ndata['wealth'] = (global_θ + m * (1-global_θ)) * (model_graph.ndata['income'] - c - i_a + (1-𝛿) * k)

def _agent_capital_past_shock_update(model_graph,model_params,timestep):
    #Applies shock from previous time step to capital carried over from previous timestep
    k,c,i_a,m = model_graph.ndata['wealth'],model_graph.ndata['wealth_consumption'],model_graph.ndata['i_a'],model_graph.ndata['m']
    global_θ = model_params['global_theta'][timestep-1]
    𝛿=model_params['depreciation']
    model_graph.ndata['wealth'] = model_graph.ndata['income'] + (global_θ + m * (1-global_θ)) + (1-𝛿) * k - c - i_a

