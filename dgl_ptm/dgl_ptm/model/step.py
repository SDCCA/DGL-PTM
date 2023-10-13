#!/usr/bin/env python
# coding: utf-8

# step - time-stepping for the poverty-trap model

from dgl_ptm.agentInteraction.trade_money import trade_money
from dgl_ptm.network.local_attachment import local_attachment 
from dgl_ptm.network.link_deletion import link_deletion 
from dgl_ptm.network.global_attachment import global_attachment
from dgl_ptm.agent.agent_update import agent_update
from dgl_ptm.model.data_collection import data_collection
from dgl_ptm.agentInteraction.weight_update import weight_update

def ptm_step(agent_graph, timestep, params):
    '''
        step - time-stepping module for the poverty-trap model

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            timestep: Current time step
            params: List of user-defined parameters

        Output:
            agent_graph: Updated agent_graph after one step of functional manipulation
    '''

    #Wealth transfer
    trade_money(agent_graph, method = params['wealth_method'])
    
    #Link/edge manipulation
    local_attachment(agent_graph, n_FoF_links = 1, edge_prop = 'weight', p_attach=1.  )
    link_deletion(agent_graph, deletion_prob = params['deletion_prob'])
    global_attachment(agent_graph, ratio = params['ratio'])
    
    #Update agent states
    agent_update(agent_graph)

    #Weight update
    weight_update(agent_graph, a = params['homophily_parameter'], b = params['characteristic_distance'],truncation_weight = params['truncation_weight'])

    #Data collection and storage
    data_collection(agent_graph, timestep = timestep, npath = params['npath'], epath = params['epath'], ndata = params['ndata'], 
                    edata = params['edata'], mode = params['mode'])
    