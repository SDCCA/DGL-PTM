import torch
import numpy as np

def income_generation(model_graph, params=None, method='pseudo_income_generation'):
    # Calculate income generated
    if method == 'pseudo_income_generation':
        _pseudo_income_generator(model_graph)    
    elif method == 'income_generation':
        _income_generator(model_graph, params)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'pseudo_income_generation' or 'income_generation'")
    

def _pseudo_income_generator(model_graph):
    TechTable =  np.array([[0.3,0],[0.35,0.15],[0.45, 0.65]])

    model_graph.ndata['income'] = torch.max((model_graph.ndata['alpha'][:,None]*model_graph.ndata['wealth'][:,None]**TechTable[:,0] - TechTable[:,1]), axis=1)[0].to(torch.float32)

def _income_generator(model_graph, params):
    gamma = params['tech_gamma']
    cost = params['tech_cost']
    model_graph.ndata['income'],model_graph.ndata['tech_index'] = torch.max((model_graph.ndata['alpha'][:,None]*model_graph.ndata['wealth'][:,None]**gamma - cost), axis=1)
