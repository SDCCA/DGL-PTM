import torch
import numpy as np

def income_generation(model_graph, device, params=None, method='default'):
    # Calculate income generated   
    if method == 'default':
        _agent_income_generator(model_graph, device, params)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'default'")
    

def _agent_income_generator(model_graph, device, params):
    gamma = params['tech_gamma'].to(device)
    cost = params['tech_cost'].to(device)
    model_graph.ndata['income'],model_graph.ndata['tech_index'] = torch.max((model_graph.ndata['alpha'][:,None]*model_graph.ndata['wealth'][:,None]**gamma - cost), axis=1)
    # TODO: declare variable alpha somewhere   
