from dgl_ptm.agent.income_generation import income_generation
from dgl_ptm.agent.wealth_consumption import wealth_consumption
from dgl_ptm.agent.agent_perception import agent_perception_update
from dgl_ptm.agent.capital_update import capital_update
import torch

def agent_update(model_graph, device, model_params, model_data=None, timestep=None, method='default'):
    '''
    agent_update - Updates agent attributes
    '''
    if method == 'capital':
        capital_update(model_graph, model_params, model_data, timestep, method=model_params['capital_update_method'])
    elif method == 'theta':
        agent_perception_update(model_graph, model_data, timestep, method=model_params['perception_method'])
    elif method == 'consumption':
        wealth_consumption(model_graph, model_params,method=model_params['consume_method'])
    elif method == 'income':
        # device is passed to this function as variables need to be moved from cpu to gpu within function
        # depending on user choice of device.
        income_generation(model_graph, device, model_params,method=model_params['income_method'])
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be one of: 'capital', 'theta', 'consumption', or 'income'")

