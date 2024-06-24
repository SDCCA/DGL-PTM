import torch
import numpy as np
from scipy.interpolate import interp1d
from scipy.optimize import minimize_scalar


def wealth_consumption(model_graph, model_params=None, method='default'):
    # Calculate wealth consumed
    if method == 'default':
        _fitted_agent_wealth_consumption(model_graph)
    elif method == 'bellman_consumption':
        _bellman_agent_wealth_consumption(model_graph,model_params)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'default', or 'bellman_consumption'")


def _fitted_agent_wealth_consumption(model_graph):
    # ToDo: Add a reference for the quation, see #83
    model_graph.ndata['wealth_consumption'] = 0.64036047*torch.log(model_graph.ndata['wealth'])

def _bellman_agent_wealth_consumption(model_graph, model_params):
    # ToDo: Implement bellman wealth model
    raise NotImplementedError(
            "Bellman wealth consumption model is currently not available. Please await future updates.")
