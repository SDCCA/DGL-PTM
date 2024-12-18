#!/usr/bin/env python

"""step - time-stepping for the poverty-trap model."""

from dgl_ptm.agent.agent_update import agent_update
from dgl_ptm.agentInteraction.trade_money import trade_money
from dgl_ptm.agentInteraction.weight_update import multi_property_weight_update
from dgl_ptm.model.data_collection import data_collection
from dgl_ptm.network.global_attachment import global_attachment
from dgl_ptm.network.link_deletion import link_deletion
from dgl_ptm.network.local_attachment import local_attachment
from dgl_ptm.network.local_attachment_basic_homophily import local_attachment_homophily
from dgl_ptm.network.random_edge_noise import random_edge_noise
from dgl_ptm.util.utils import sample_distribution_tensor

def ptm_step(agent_graph, device, timestep, params):
    """Step - time-stepping module for the poverty-trap model.

    Args:
        agent_graph: DGLGraph with agent nodes and edges connecting agents
        device: Device to run the model on, e.g. 'cpu' or 'cuda'
        timestep: Current time step
        params: List of user-defined parameters

    Output:
        agent_graph: Updated agent_graph after one step of functional manipulation
    """
    if params['step_type']=='default':
        #Wealth transfer
        trade_money(agent_graph, device, method = params['trade_method'])

        #Edge manipulation
        local_attachment(
            agent_graph, n_FoF_links = 1, edge_prop = 'weight', p_attach=1.
            )
        link_deletion(
            agent_graph,
            method = params['del_method'],
            threshold = params['del_threshold']
            )
        global_attachment(agent_graph, device, ratio = params['noise_ratio'])

        #Update agent states
        agent_update(agent_graph, params, device=device)

        #Weight update
        multi_property_weight_update(
            agent_graph,
            device,
            truncation_weight = params['truncation_weight'],
            properties = {'wealth':{'keys':["wealth"],'homophily_parameter':params['homophily_parameter'],'characteristic_distance':params['characteristic_distance']}})
        
    elif params['step_type']=='ptm':
        if timestep==0:
            if agent_graph.number_of_edges()+params['noise_ratio']*agent_graph.number_of_nodes()+params['local_ratio']*agent_graph.number_of_nodes()<2**32:
                agent_graph = agent_graph.int()
                print(f"Agent graph storage type: {agent_graph.idtype}")



            #Update agent income
            agent_update(agent_graph,
                         params,
                         device=device,
                         method ='income'
                         )
            #Update agent consumption
            agent_update(
                agent_graph,
                params,
                device=device,
                timestep=timestep,
                method ='consumption'
                )
            #Collect specified data
            data_collection(
                agent_graph,
                timestep = timestep,
                npath = params['npath'],
                epath = params['epath'],
                ndata = params['ndata'],
                edata = params['edata'],
                mode = params['mode']
                )
            return
        #For timestep 1 and beyond:

        #Update agent capital, k_t+1 for the previous step becomes k_t
        agent_update(
            agent_graph, 
            params, 
            device=device, 
            timestep=timestep, 
            method = 'capital'
            )
        #Update agent theta with the information from the previous step
        agent_update(
            agent_graph, params, device=device, timestep=timestep-1, method ='theta'
            )

        #Update edge weights

        multi_property_weight_update(
            agent_graph,
            device,
            truncation_weight = params['truncation_weight'],
            properties = params['homophily_basis'])


        #Edge manipulation
        start_edges = agent_graph.number_of_edges()
        random_edge_noise(
            agent_graph,
            device,
            n_perturbances = int(params['noise_ratio']*agent_graph.number_of_nodes())
            )
        local_attachment_homophily(
            agent_graph,
            device,
            n_FoF_links = int(params['local_ratio']*agent_graph.number_of_nodes()),
            homophily_parameter = params['homophily_parameter'],
            characteristic_distance = params['characteristic_distance'],
            truncation_weight = params['truncation_weight']
            )
        if params['del_threshold'] == 'balance':
            threshold = int((agent_graph.number_of_edges()-start_edges)/2)
        else:
            threshold = params['del_threshold']
        link_deletion(
            agent_graph, method = params['del_method'], threshold = threshold
            )
        #Update agent degree and weighted degree
        agent_update(agent_graph, method='degree')
        agent_update(agent_graph, method='weighted_degree')

        #Wealth transfer
        trade_money(agent_graph, device, method = params['trade_method'])


        # Update agent income
        agent_update(agent_graph, params, device=device, method ='income')
        # Predict agent consumption (and investment if applicable)
        agent_update(
            agent_graph, params, device=device, timestep=timestep, method ='consumption'
            )


    # Data can be collected periodically (every X steps) and/or at specified time steps.
    do_periodical_data_collection = (
        params['data_collection_period'] > 0
        and timestep % params['data_collection_period'] == 0
        )
    do_specific_data_collection = (
        params['data_collection_list']
        and timestep in params['data_collection_list']
        )
    if do_periodical_data_collection or do_specific_data_collection:
        #Data collection and storage
        data_collection(
            agent_graph,
            timestep = timestep,
            npath = params['npath'],
            epath = params['epath'],
            ndata = params['ndata'],
            edata = params['edata'],
            mode = params['mode']
            )

def sveir_step(agent_graph, device, timestep, params):
    """Step - time-stepping module for the SVEIR model.

    Args:
        agent_graph: DGLGraph with agent nodes and edges connecting agents
        device: Device to run the model on, e.g. 'cpu' or 'cuda'
        timestep: Current time step
        params: List of user-defined parameters

    Output:
        agent_graph: Updated agent_graph after one step of functional manipulation
    """
    # map from compartment to index
    m = {
        "S":0,
        "V":1,
        "E":2,
        "I":3,
        "R":4
    }

    # random tensors
    bounds = [0.0, 1.0]
    num_nodes = agent_graph.num_nodes()
    i_r_rng = sample_distribution_tensor('uniform', bounds, num_nodes)
    s_r_rng = sample_distribution_tensor('uniform', bounds, num_nodes)
    s_e_rng = sample_distribution_tensor('uniform', bounds, num_nodes)
    v_e_rng = sample_distribution_tensor('uniform', bounds, num_nodes)

    # increment exposure time
    agent_graph.ndata["exposure_time"][agent_graph.ndata["compartments"]==m["E"]] += 1

    # transition from Exposed to Infectious
    exposed_to_infections = (agent_graph.ndata["compartments"]==m["E"]) & (agent_graph.ndata["exposure_time"] >= params["exposure_period"])
    agent_graph.ndata["compartments"][exposed_to_infections] = m["I"]

    # transition from Infectious to Recovered
    infectious_to_recovered = (agent_graph.ndata["compartments"]==m["I"]) & (i_r_rng < params["recovery_rate"])
    agent_graph.ndata["compartments"][infectious_to_recovered] = m["R"]

    # transition from Susceptible to Vaccinated
    susceptible_to_vaccinated = (agent_graph.ndata["compartments"]==m["S"]) & (s_r_rng < params["vaccination_rate"])
    agent_graph.ndata["compartments"][susceptible_to_vaccinated] = m["V"]

    # Transition from Susceptible to Exposed
    susceptible_nodes = (agent_graph.ndata["compartments"] == m["S"]).nonzero(as_tuple=True)[0]
    for node in susceptible_nodes:
        neighbors = agent_graph.successors(node)
        infected_neighbors = neighbors[agent_graph.ndata["compartments"][neighbors] == m["I"]]
        if infected_neighbors.numel() > 0 and s_e_rng[node] < params["infection_probability"]:
            agent_graph.ndata["compartments"][node] = m["E"]
            agent_graph.ndata["exposure_time"][node] = 0

    # Transition from Vaccinated to Exposed
    vaccinated_nodes = (agent_graph.ndata["compartments"] == m["V"]).nonzero(as_tuple=True)[0]
    for node in vaccinated_nodes:
        neighbors = agent_graph.successors(node)
        infected_neighbors = neighbors[agent_graph.ndata["compartments"][neighbors] == m["I"]]
        if infected_neighbors.numel() > 0 and v_e_rng[node] < params["infection_probability"] * (1 - params["vaccine_efficacy"]):
            agent_graph.ndata["compartments"][node] = m["E"]
            agent_graph.ndata["exposure_time"][node] = 0

    # Data can be collected periodically (every X steps) and/or at specified time steps.
    do_periodical_data_collection = (
        params['data_collection_period'] > 0
        and timestep % params['data_collection_period'] == 0
        )
    do_specific_data_collection = (
        params['data_collection_list']
        and timestep in params['data_collection_list']
        )
    if do_periodical_data_collection or do_specific_data_collection:
        data_collection(
            agent_graph,
            timestep = timestep,
            npath = params['npath'],
            epath = params['epath'],
            ndata = params['ndata'],
            edata = params['edata'],
            mode = params['mode']
        )
