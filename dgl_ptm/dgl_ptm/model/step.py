#!/usr/bin/env python

"""step - time-stepping for the poverty-trap model."""

from dgl_ptm.agent.agent_update import agent_update
from dgl_ptm.agentInteraction.trade_money import trade_money
from dgl_ptm.agentInteraction.weight_update import weight_update
from dgl_ptm.model.data_collection import data_collection
from dgl_ptm.network.global_attachment import global_attachment
from dgl_ptm.network.link_deletion import link_deletion
from dgl_ptm.network.local_attachment import local_attachment
from dgl_ptm.network.local_attachment_basic_homophily import local_attachment_homophily
from dgl_ptm.network.random_edge_noise import random_edge_noise


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
        trade_money(agent_graph, device, method = params['wealth_method'])

        #Link/edge manipulation
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
        weight_update(
            agent_graph,
            device,
            homophily_parameter = params['homophily_parameter'],
            characteristic_distance = params['characteristic_distance'],
            truncation_weight = params['truncation_weight']
            )

    elif params['step_type']=='custom':
        if timestep==0:
            #Update agent states
            agent_update(
                agent_graph,
                params,
                device=device,
                timestep=timestep,
                method ='theta'
                )
            agent_update(agent_graph,
                         params,
                         device=device,
                         method ='income'
                         )
            agent_update(
                agent_graph,
                params,
                device=device,
                timestep=timestep,
                method ='consumption'
                )
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

        agent_update(
            agent_graph, params, device=device, timestep=timestep, method = 'capital'
            )

        #Weight update
        weight_update(
            agent_graph,
            device,
            homophily_parameter = params['homophily_parameter'],
            characteristic_distance = params['characteristic_distance'],
            truncation_weight = params['truncation_weight']
            )

        #Link/edge manipulation
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
        threshold = int((agent_graph.number_of_edges()-start_edges)/2)
        link_deletion(
            agent_graph, method = params['del_method'], threshold = threshold
            )

        #Wealth transfer
        trade_money(agent_graph, device, method = params['wealth_method'])

        #Update agent states
        agent_update(
            agent_graph, params, device=device, timestep=timestep, method ='theta'
            )
        agent_update(agent_graph, params, device=device, method ='income')
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
