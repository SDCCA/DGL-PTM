import torch
import numpy as np


def income_generation(model_graph, device, params=None, method='pseudo_income_generation'):
    # Calculate income generated
    if method == 'pseudo_income_generation':
        _pseudo_income_generator(model_graph)    
    elif method == 'income_generation':
        _income_generator(model_graph,device, params)
    else:
        raise NotImplementedError("Incorrect income generation method received.)
    

def _pseudo_income_generator(model_graph):
    TechTable =  np.array([[0.3,0],[0.35,0.15],[0.45, 0.65]])

    model_graph.ndata['income'] = torch.max((model_graph.ndata['alpha'][:,None]*model_graph.ndata['wealth'][:,None]**TechTable[:,0] - TechTable[:,1]), axis=1)[0].to(torch.float32)

def _income_generator(model_graph, device, params):
    gamma = params['tech_gamma'].to(device)
    cost = params['tech_cost'].to(device)
    model_graph.ndata['income'],model_graph.ndata['tech_index'] = torch.max((model_graph.ndata['alpha'][:,None]*model_graph.ndata['wealth'][:,None]**gamma - cost), axis=1)
    # TODO: declare variable alpha somewhere   
