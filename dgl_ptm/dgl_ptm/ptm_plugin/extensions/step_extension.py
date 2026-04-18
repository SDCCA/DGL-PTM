"""This module contains the step function for the poverty-trap model.

Function(s):
- ptm_step: Time-stepping module for the poverty-trap model
"""

import pluggy

hookimpl = pluggy.HookimplMarker("dgl_abm")

from dgl_abm.agent.agent_update import agent_update
from dgl_ptm.ptm_plugin.agentInteraction.exchange_capital import exchange_capital
from dgl_ptm.ptm_plugin.agentInteraction.weight_update import weight_update
from dgl_ptm.ptm_plugin.agentInteraction.weight_update import multi_property_weight_update
from dgl_abm.model.data_collection import data_collection
from dgl_abm.network.global_attachment import global_attachment
from dgl_abm.network.link_deletion import link_deletion
from dgl_abm.network.local_attachment import local_attachment
from dgl_abm.network.local_attachment_basic_homophily import local_attachment_homophily
from dgl_abm.network.random_edge_noise import random_edge_noise


@hookimpl
def custom_steps():
    return {
        "ptm": _ptm_step
    }

def _ptm_step(agent_graph, device, timestep, params):
    """Execute step for the poverty-trap model.

    Args:
        agent_graph (DGLGraph): All agent node and edge data
        device (torch.device): Device on which to perform computations
        timestep (int): Current time step
        params (dict): Parameters specified in model configuration and initialization

    Effects: Completes updates to agent_graph after one step of functional manipulation
    """
    if params['step_type']=='default':
        #Wealth transfer
        exchange_capital(agent_graph, device, method = params['trade_method'])

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
            if (agent_graph.number_of_edges()+params['noise_ratio'] * 
                agent_graph.number_of_nodes()+params['local_ratio'] *
                agent_graph.number_of_nodes()<2**32):
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
            n_links = int(params['local_ratio']*agent_graph.number_of_nodes()),
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
        exchange_capital(agent_graph, device, method = params['trade_method'])


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
