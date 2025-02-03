from dgl_ptm.agent.income_generation import income_generation
from dgl_ptm.agent.wealth_consumption import wealth_consumption
from dgl_ptm.agent.capital_update import capital_update
from dgl_ptm.util.network_metrics import node_degree, node_weighted_degree
from dgl_ptm.util.utils import sample_distribution_tensor
import torch
#from dgl_ptm.agent.movement import move_agents


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
    elif method == 'position':
        _agent_position_update(model_graph)
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
    global_θ =model_params['global_theta'][timestep]
    model_graph.ndata['theta'] = model_graph.ndata['theta'] * (1-model_graph.ndata['sensitivity']) + global_θ * model_graph.ndata['sensitivity']

def _agent_consumption_update(model_graph, model_params, timestep, device):
    '''Updates agent consumption based on method specified in model parameters.'''
    wealth_consumption(model_graph, model_params,timestep, device, method=model_params['consume_method'])

def _agent_income_update(model_graph, model_params, device):
    '''Updates agent income based on method specified in model parameters.'''
    income_generation(model_graph,device,model_params,method=model_params['income_method'])

def _agent_degree_update(model_graph):
    '''Updates agent degree. Note both edge directions are considered.'''
    model_graph.ndata['degree'] = node_degree(model_graph)

def _agent_weighted_degree_update(model_graph):
    '''Updates agent weighted degree. Note both edge directions are considered.'''
    model_graph.ndata['weighted_degree'] = node_weighted_degree(model_graph)

def _agent_position_update(model_graph,model_params,moving_agents):
    '''Updates agent position.'''
    move_agents(model_graph,model_params,moving_agents)


def sveir_agent_update(method, agent_graph, M, params=None, num_nodes=None, edge_weights=None):
    if method == "exposure_increment":
        _agent_increment_exposure_time(agent_graph, M)
    elif method == "exposed_to_infectious":
        _agent_exposed_to_infectious(agent_graph, M, params)
    elif method == "infectious_to_recovered":
        _agent_infectious_to_recovered(agent_graph, M, params, num_nodes)
    elif method == "susceptible_to_vaccinated":
        _agent_susceptible_to_vaccinated(agent_graph, M, params, num_nodes)
    elif method == "susceptible_to_exposed":
        _agent_susceptible_to_exposed(agent_graph, M, params, num_nodes, edge_weights)
    elif method == "vaccinated_to_exposed":
        _agent_vaccinated_to_exposed(agent_graph, M, params, num_nodes, edge_weights)

def _agent_increment_exposure_time(agent_graph, M):
    agent_graph.ndata["exposure_time"][agent_graph.ndata["compartments"] == M["E"]] += 1

def _agent_exposed_to_infectious(agent_graph, M, params):
    exposed_to_infections = (agent_graph.ndata["compartments"]==M["E"]) & (agent_graph.ndata["exposure_time"] >= params["exposure_period"])
    agent_graph.ndata["compartments"][exposed_to_infections] = M["I"]
    agent_graph.ndata["num_infections"][exposed_to_infections] += 1

def _agent_infectious_to_recovered(agent_graph, M, params, num_nodes):
    i_r_rng = sample_distribution_tensor('uniform', [0.0, 1.0], num_nodes)
    infectious_to_recovered = (agent_graph.ndata["compartments"]==M["I"]) & (i_r_rng < params["recovery_rate"])
    agent_graph.ndata["compartments"][infectious_to_recovered] = M["R"]

def _agent_susceptible_to_vaccinated(agent_graph, M, params, num_nodes):
    s_r_rng = sample_distribution_tensor('uniform', [0.0, 1.0], num_nodes)
    susceptible_to_vaccinated = (agent_graph.ndata["compartments"]==M["S"]) & (s_r_rng < params["vaccination_rate"])
    agent_graph.ndata["compartments"][susceptible_to_vaccinated] = M["V"]

def _agent_susceptible_to_exposed(agent_graph, M, params, num_nodes, edge_weights):
    susceptible_recovered_nodes = torch.where((agent_graph.ndata["compartments"] == M["S"]) | (agent_graph.ndata["compartments"] == M["R"]))[0]
    infected_weights = torch.where(agent_graph.ndata["compartments"].repeat(num_nodes, 1) == 3, edge_weights, 0)[susceptible_recovered_nodes]
    nonzero_weights = torch.where(infected_weights > 0)

    RNG = torch.rand((2, nonzero_weights[0].shape[0]))
    RNG1 = torch.zeros_like(infected_weights)
    RNG1[nonzero_weights] = RNG[0]
    RNG2 = torch.zeros_like(infected_weights)
    RNG2[nonzero_weights] = RNG[1]

    prob_infection = params["infection_probability"] * torch.exp(-1.5 * agent_graph.ndata["num_infections"][susceptible_recovered_nodes])
    infection = (infected_weights > 0).type(torch.float32) * (RNG1 < infected_weights) * (RNG2 < prob_infection[:, None])
    infected_nodes = susceptible_recovered_nodes[torch.where(infection.sum(axis=1) > 0)]

    agent_graph.ndata["compartments"][infected_nodes] = M["E"]
    agent_graph.ndata["exposure_time"][infected_nodes] = 0

def _agent_vaccinated_to_exposed(agent_graph, M, params, num_nodes, edge_weights):
    vaccinated_nodes = (agent_graph.ndata["compartments"] == M["V"]).nonzero(as_tuple=True)[0]
    infected_weights = torch.where(agent_graph.ndata["compartments"].repeat(num_nodes, 1) == 3, edge_weights, 0)[vaccinated_nodes]
    nonzero_weights = torch.where(infected_weights > 0)

    RNG = torch.rand((2, nonzero_weights[0].shape[0]))
    RNG1 = torch.zeros_like(infected_weights)
    RNG1[nonzero_weights] = RNG[0]
    RNG2 = torch.zeros_like(infected_weights)
    RNG2[nonzero_weights] = RNG[1]

    prob_infection = (1-params["vaccine_efficacy"]) * params["infection_probability"] * torch.exp(-1.5 * agent_graph.ndata["num_infections"][vaccinated_nodes])
    infection = (infected_weights > 0).type(torch.float32) * (RNG1 < infected_weights) * (RNG2 < prob_infection[:, None])
    infected_nodes = vaccinated_nodes[torch.where(infection.sum(axis=1) > 0)]

    agent_graph.ndata["compartments"][infected_nodes] = M["E"]
    agent_graph.ndata["exposure_time"][infected_nodes] = 0
