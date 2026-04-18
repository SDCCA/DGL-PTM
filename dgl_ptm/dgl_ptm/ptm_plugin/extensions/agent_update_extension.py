import pluggy

hookimpl = pluggy.HookimplMarker("dgl_abm")

from dgl_ptm.ptm_plugin.agent.capital_update import capital_update
from dgl_ptm.ptm_plugin.agent.wealth_consumption import wealth_consumption
from dgl_ptm.ptm_plugin.agent.income_generation import income_generation

@hookimpl
def agent_update_methods():
    return {
        "capital": _agent_capital_update,
        "theta": _agent_theta_update,
        "consumption": _agent_consumption_update,
        "income": _agent_income_update,
    }

def _agent_capital_update(model_graph,model_params,timestep):
    """Update agent capital based on method specified in model parameters.

    Note: k_t+1 becomes the new k_t

    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time

    Returns:
        None
    """
    capital_update(model_graph, model_params, timestep, 
                   method=model_params['capital_method'])

    
def _agent_theta_update(model_graph,model_params,timestep):
    """Update agent perception of theta based on observation and sensitivity.

    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time

    Returns:
        None
    """
    global_θ =model_params['global_theta'][timestep]
    model_graph.ndata['theta'] = (model_graph.ndata['theta'] * 
                                  (1-model_graph.ndata['sensitivity']) + 
                                  global_θ * model_graph.ndata['sensitivity'])

def _agent_consumption_update(model_graph, model_params, timestep, device):
    """Update agent consumption based on method specified in model parameters.
    
    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration/initialization
        timestep (int): Current model time
        device (torch.device): Device on which to perform computations

    Returns:
        None
    """
    wealth_consumption(model_graph, model_params,timestep, device, 
                       method=model_params['consume_method'])

def _agent_income_update(model_graph, model_params, device):
    """Update agent income based on method specified in model parameters.
    
    Args:
        model_graph (DGLGraph): All agent data
        model_params (dict): Parameters specified in model configuration/initialization
        device (torch.device): Device on which to perform computations
    
    Returns:
        None
    """
    income_generation(model_graph,device,model_params,
                      method=model_params['income_method'])
