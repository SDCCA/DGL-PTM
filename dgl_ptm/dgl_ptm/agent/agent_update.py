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


def sveir_agent_update(method, agent_graph, M=None, params=None, num_nodes=None, edge_weights=None, grid=None, adjacency=None, random_activity=None):
    if method == "exposure_increment":
        _agent_increment_exposure_time(agent_graph, M)
    elif method == "exposed_to_infectious":
        _agent_exposed_to_infectious(agent_graph, M, params)
    elif method == "infectious_to_recovered":
        _agent_infectious_to_recovered(agent_graph, M, params, num_nodes)
    elif method == "susceptible_to_vaccinated":
        _agent_susceptible_to_vaccinated(agent_graph, M, params, num_nodes)
    elif method == "susceptible_to_exposed":
        _agent_susceptible_to_exposed(agent_graph, M, params, num_nodes, adjacency)
    elif method == "vaccinated_to_exposed":
        _agent_vaccinated_to_exposed(agent_graph, M, params, num_nodes, adjacency)
    elif method == "move":
        return _agent_move(agent_graph, edge_weights)
    elif method == "human_to_water_transmission":
        _agent_human_to_water_transmission(agent_graph, M, params, grid, random_activity)
    elif method == "water_to_human_transmission":
        _agent_water_to_human_transmission(agent_graph, M, params, grid, random_activity)
    elif method == "water_recovery":
        _water_recovery(params, grid)

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

def _agent_susceptible_to_exposed(agent_graph, M, params, num_nodes, adjacency):
    susceptible_recovered_nodes = torch.where((agent_graph.ndata["compartments"] == M["S"]) | (agent_graph.ndata["compartments"] == M["R"]))[0]
    infected_weights = torch.where(agent_graph.ndata["compartments"].repeat(num_nodes, 1) == 3, adjacency, 0)[susceptible_recovered_nodes]
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

def _agent_vaccinated_to_exposed(agent_graph, M, params, num_nodes, adjacency):
    vaccinated_nodes = (agent_graph.ndata["compartments"] == M["V"]).nonzero(as_tuple=True)[0]
    infected_weights = torch.where(agent_graph.ndata["compartments"].repeat(num_nodes, 1) == 3, adjacency, 0)[vaccinated_nodes]
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

def _agent_move(agent_graph, edge_weights):
    random_activity = torch.multinomial(agent_graph.ndata["time_use"], num_samples=1).squeeze()
    agent_graph.ndata["activity_choice"] = random_activity

    # 0 -> home
    agents_home = torch.where(random_activity==0)[0]
    agent_graph.ndata['x'][agents_home] = agent_graph.ndata["home_location"][agents_home,0]
    agent_graph.ndata['y'][agents_home] = agent_graph.ndata["home_location"][agents_home,1]

    # 1 -> school
    agents_school = torch.where(random_activity==1)[0]
    agent_graph.ndata['x'][agents_school] = agent_graph.ndata["school_location"][agents_school,0]
    agent_graph.ndata['y'][agents_school] = agent_graph.ndata["school_location"][agents_school,1]

    # 2 -> place of worship
    agents_worship = torch.where(random_activity==2)[0]
    agent_graph.ndata['x'][agents_worship] = agent_graph.ndata["worship_location"][agents_worship,0]
    agent_graph.ndata['y'][agents_worship] = agent_graph.ndata["worship_location"][agents_worship,1]

    # 3 -> water collection, water contamination by human, human contamination from water
    agents_water = torch.where(random_activity==3)[0]
    agent_graph.ndata['x'][agents_water] = agent_graph.ndata["water_location"][agents_water,0]
    agent_graph.ndata['y'][agents_water] = agent_graph.ndata["water_location"][agents_water,1]

    # 4 -> social
    agents_social = torch.where(random_activity==4)[0]
    social_weights = edge_weights[agents_social]
    # social agents can only visit agents that are at home
    at_home_mask = torch.zeros(social_weights.shape[1], dtype=torch.bool)
    at_home_mask[agents_home] = True
    social_weights = social_weights * at_home_mask
    # only consider social agents that have at least one available neighbor
    non_zero_indices = torch.where(torch.sum(social_weights, dim=1) != 0)[0]
    agents_social = agents_social[non_zero_indices]
    social_weights = social_weights[non_zero_indices]
    # probabilities of visiting agents that are at home must sum to one
    social_weights = social_weights / social_weights.sum(dim=1, keepdim=True)
    # identify which neighbors to visit and update social agents' locations
    visit_indices = torch.multinomial(social_weights, num_samples=1).squeeze()
    agent_graph.ndata['x'][agents_social] = agent_graph.ndata["home_location"][visit_indices,0]
    agent_graph.ndata['y'][agents_social] = agent_graph.ndata["home_location"][visit_indices,1]

    return random_activity

def _agent_water_to_human_transmission(agent_graph, M, params, grid, random_activity):
    infected_water_coords = torch.stack(torch.where(grid.get_slice("water")==2)).T
    if infected_water_coords.shape[0] == 0:
        return
    
    RNG = torch.rand(random_activity.shape[0])

    coords = torch.stack((agent_graph.ndata["x"], agent_graph.ndata["y"])).T
    match_agent_coords_infected_water_coords = (coords[:, None, :] == infected_water_coords).all(dim=2)
    agents_collecting_infected_water = match_agent_coords_infected_water_coords.any(dim=1)

    agents_susceptible = agent_graph.ndata["compartments"] == M["S"]
    agents_recovered = agent_graph.ndata["compartments"] == M["R"]
    agents_vaccinated = agent_graph.ndata["compartments"] == M["V"]

    s_r_agents = (agents_susceptible | agents_recovered) & (agents_collecting_infected_water)
    prob_infection_s_r = params["water_to_human_infection_prob"] * torch.exp(-1.5 * agent_graph.ndata["num_infections"])
    s_r_infection = torch.where(s_r_agents, RNG, 1) < prob_infection_s_r
    s_r_infected_nodes = torch.where(s_r_infection)[0]
    agent_graph.ndata["compartments"][s_r_infected_nodes] = M["E"]
    agent_graph.ndata["exposure_time"][s_r_infected_nodes] = 0

    v_agents = (agents_vaccinated) & (agents_collecting_infected_water)
    prob_infection_v = (1-params["vaccine_efficacy"]) * params["water_to_human_infection_prob"] * torch.exp(-1.5 * agent_graph.ndata["num_infections"])
    v_infection = torch.where(v_agents, RNG, 1) < prob_infection_v
    v_infected_nodes = torch.where(v_infection)[0]
    agent_graph.ndata["compartments"][v_infected_nodes] = M["E"]
    agent_graph.ndata["exposure_time"][v_infected_nodes] = 0

def _agent_human_to_water_transmission(agent_graph, M, params, grid, random_activity):

    water_slice = grid.get_slice("water")
    non_infected_water_coords = torch.stack(torch.where(water_slice==1)).T
    if non_infected_water_coords.shape[0] == 0:
        return
    
    RNG = torch.rand(random_activity.shape[0])
    
    agents_collecting_water = random_activity == 3
    infected_agents = agent_graph.ndata["compartments"]==M["I"]

    agents_capable_of_infecting_water = (agents_collecting_water) & (infected_agents)
    agents_infecting_water = torch.where(agents_capable_of_infecting_water, RNG, 1) < params["human_to_water_infection_prob"]

    water_points_to_infect = torch.unique(agent_graph.ndata["water_location"][agents_infecting_water], dim=0).int()
    if water_points_to_infect.shape[0] == 0:
        return

    water_slice[water_points_to_infect[:,0], water_points_to_infect[:,1]] = 2

def _water_recovery(params, grid):
    water_slice = grid.get_slice("water")
    infected_water_coords = torch.stack(torch.where(water_slice==2)).T
    if infected_water_coords.shape[0] == 0:
        return

    RNG = torch.rand(infected_water_coords.shape[0], 1)
    recovery = RNG < params["water_recovery_prob"]
    recovered_coords = infected_water_coords[torch.where(recovery)[0]]
    if recovered_coords.shape[0] == 0:
        return

    water_slice[recovered_coords[:,0], recovered_coords[:,1]] = 1
