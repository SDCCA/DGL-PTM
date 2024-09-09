from dgl_ptm.agent.income_generation import income_generation
from dgl_ptm.agent.wealth_consumption import wealth_consumption
from dgl_ptm.agent.capital_update import capital_update
from dgl_ptm.util.network_metrics import node_degree, node_weighted_degree


def agent_update(model_graph, model_params=None, device=None, timestep=None, method='pseudo'):
    '''
    agent_update - Updates agent attributes
    '''
    if method == 'capital':
        _agent_capital_update(model_graph, model_params, timestep)
    elif method == 'theta':
        _agent_theta_update(model_graph, model_params, timestep)
    elif method == 'consumption':
        _agent_consumption_update(model_graph, model_params, timestep, device)
    elif method == 'income':
        _agent_income_update(model_graph,model_params,device)
    elif method == 'degree':
        _agent_degree_update(model_graph)
    elif method == 'weighted_degree':
        _agent_weighted_degree_update(model_graph)
    elif method == 'pseudo':
        _pseudo_agent_update(model_graph,model_params,device)
    else:
        raise NotImplementedError(f"Unrecognized agent update type {method} attempted during time step implementation.'")

def _pseudo_agent_update(model_graph,model_params,device): 
    '''
    agent_update - Updates the state of the agent based on income generation and money trades
    '''
    model_graph.ndata['wealth'] = model_graph.ndata['wealth'] + model_graph.ndata['net_trade']
    income_generation(model_graph, device, model_params, method = model_params['income_method'])
    wealth_consumption(model_graph, model_params, method=model_params['consume_method'], device=device)
    model_graph.ndata['wealth'] = model_graph.ndata['wealth'] + model_graph.ndata['income'] - model_graph.ndata['wealth_consumption']



def _agent_capital_update(model_graph,model_params,timestep):
    '''
    formula for k_t+1 is applied at the beginning of each time step 
    k_t+1 becomes the new k_t
    '''
    capital_update(model_graph, model_params, timestep, method=model_params['capital_method'])
    #self.connections=0
    #self.trades=0
    #self.net_traded=model_graph.ndata['wealth']
    
def _agent_theta_update(model_graph,model_params,timestep):
    '''Updates agent perception of theta based on observation and sensitivity'''
    global_θ =model_params['model_theta'][timestep]
    model_graph.ndata['theta'] = model_graph.ndata['theta'] * (1-model_graph.ndata['sensitivity']) + global_θ * model_graph.ndata['sensitivity']

def _agent_consumption_update(model_graph, model_params, timestep, device):
    '''Updates agent consumption based on method specified in model parameters.'''
    wealth_consumption(model_graph, model_params,timestep, device, method=model_params['consume_method'])

def _agent_income_update(model_graph, model_params, device):
    '''Updates agent income based on method specified in model parameters.'''
    income_generation(model_graph,device,model_params,method=model_params['income_method'])

def _agent_degree_update(model_graph):
    '''Updates agent degree. Note both directions are considered.'''
    model_graph.ndata['degree'] = node_degree(model_graph)

def _agent_weighted_degree_update(model_graph):
    '''Updates agent weighted degree. Note both directions are considered.'''
    model_graph.ndata['weighted_degree'] = node_weighted_degree(model_graph)



