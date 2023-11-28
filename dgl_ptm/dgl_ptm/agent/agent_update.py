from dgl_ptm.agent.income_generation import income_generation
from dgl_ptm.agent.wealth_consumption import wealth_consumption
import torch

def agent_update(model_graph, model_params, timestep=None, method='pseudo'):
    '''
    agent_update - Updates agent attributes
    '''
    if method=='pseudo':
        _pseudo_agent_update(model_graph,model_params)
    elif method == 'capital':
        _agent_capital_update(model_graph,model_params,timestep)
    elif method == 'theta':
        _agent_theta_update(model_graph,model_params,timestep)
    elif method == 'consumption':
        _agent_consumption_update(model_graph,model_params)
    elif method == 'income':
        _agent_income_update(model_graph,model_params)
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be 'pseudo_consumption' or 'optimized_wealth_consumption'")

def _pseudo_agent_update(model_graph,model_params): 
    '''
    agent_update - Updates the state of the agent based on income generation and money trades
    '''
    model_graph.ndata['wealth'] = model_graph.ndata['wealth'] + model_graph.ndata['net_trade']
    income_generation(model_graph, model_params, method = model_params['income_method'])
    wealth_consumption(model_graph, method=model_params['consume_method'])
    model_graph.ndata['wealth'] = model_graph.ndata['wealth'] + model_graph.ndata['income'] - model_graph.ndata['wealth_consumption']



def _agent_capital_update(model_graph,model_params,timestep):
    
    
    #formula for k_t+1 is applied at the beginning of each time step 
    # k_t+1 becomes the new k_t
    
    k,c,i_a,m,α = model_graph.ndata['wealth'],model_graph.ndata['wealth_consumption'],model_graph.ndata['i_a'],model_graph.ndata['m'],model_graph.ndata['alpha']
    
    global_θ =model_params['modelTheta'][timestep]
    𝛿=model_params['depreciation']
    model_graph.ndata['wealth'] = (global_θ + m * (1-global_θ)) * (model_graph.ndata['income'] - c - i_a + (1-𝛿) * k)
    #self.connections=0
    #self.trades=0
    #self.net_traded=model_graph.ndata['wealth']
    
def _agent_theta_update(model_graph,model_params,timestep):
    #updates agent perception of theta based on observation and sensitivity
    global_θ =model_params['modelTheta'][timestep]
    model_graph.ndata['theta'] = model_graph.ndata['theta'] * (1-model_graph.ndata['sensitivity']) + global_θ * model_graph.ndata['sensitivity']

def _agent_consumption_update(model_graph,model_params):
    wealth_consumption(model_graph, model_params,method=model_params['consume_method'])

def _agent_income_update(model_graph,model_params):
    income_generation(model_graph,model_params,method=model_params['income_method'])
