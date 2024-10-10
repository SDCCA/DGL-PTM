import os
import torch 
import dgl 
import dgl.function as fn

def trade_money(agent_graph, device, method: str):
    """ Trades money between the different connected agents based on 
        wealth (k) and savings propensity (lambda). 
        
        Two methods are provided for wealth exchange: 
        1 - Random weighted wealth transfer to all connected neigbours
        2 - Choose one random neighbour for transfering wealth 

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            method: String with method choice of 'weighted_transfer' or 'singular_transfer'

        Output:
            agent_graph.ndata['net_trade']: Adds node attribute 'net_trade' with sum of
                capital transfered from connected agent nodes to self minus the outflow
                of capital to connected agent nodes in this time-step.

    NOTE: Assumes that the following properties are available already:
        k, lambda, w (for 'weighted_transfer'), zeros, ones, total neighbour count
    NOTE: All edges are bidirected with uniform weights 'w'

    TODO: Rename variables as per Thijs' updates on notebook
    """
    # Calculating disposable wealth
    print(f"k before:{agent_graph.ndata['wealth'][0:5]}")
    agent_graph.ndata['disposable_wealth'] = agent_graph.ndata['lambda']*agent_graph.ndata['wealth'] # TODO: declare what lambda is
    
    # Transfer of wealth
    if method == 'weighted_transfer':
        _weighted_transfer(agent_graph, device)
    elif method == 'singular_transfer':
        _singular_transfer(agent_graph, device)
    elif method == 'no_transfer':
        pass
    else:
        raise NotImplementedError("Incorrect method received. \
                         Method needs to be either 'weighted_transfer' or 'singular_transfer'")
    
def _weighted_transfer(agent_graph, device):
    """
        Weighted transfer of wealth from each agent (node) to every connected 
        neighbour agent based on pre-defined edge weights.
    """

    # Sum all incoming weights
    agent_graph.ndata['total_weight'] = torch.zeros(agent_graph.num_nodes()).to(device)
    agent_graph.update_all(fn.u_add_e('total_weight','weight','total_weight_msg'), fn.sum('total_weight_msg', 'total_weight'))

    # Calculating outgoing weight %s
    agent_graph.apply_edges(fn.e_div_u('weight','total_weight','percent_weight'))

    # Wealth transfer amount on each edge
    agent_graph.apply_edges(fn.e_mul_u('percent_weight','disposable_wealth','trfr_wealth'))  # TODO: check what trfr_wealth actually is

    # Sum total incoming wealth
    agent_graph.update_all(fn.v_add_e('zeros','trfr_wealth','net_trade_msg'), fn.sum('net_trade_msg', 'net_trade'))
    print(f"Total In:{agent_graph.ndata['net_trade'][0:5]}")

    # Subtract outgoing wealth
    agent_graph.ndata['net_trade'] = agent_graph.ndata['net_trade'] - agent_graph.ndata['disposable_wealth']
    print(f"Disposable Wealth:{agent_graph.ndata['disposable_wealth'][0:5]}")  
    print(f"Net Trade:{agent_graph.ndata['net_trade'][0:5]}")

    # Conduct exchange
    agent_graph.ndata['wealth'] = agent_graph.ndata['wealth'] + agent_graph.ndata['net_trade']
    print(f"k after:{agent_graph.ndata['wealth'][0:5]}")

def _singular_transfer(agent_graph, device):
    """
        Singular transfer of wealth from each agent (node) to one randomly
        selected, connected neighbour.
    """

    # Subsample graph to one random edge per node
    graph_subset = dgl.sampling.sample_neighbors(agent_graph, agent_graph.nodes(), 1, edge_dir='out', copy_ndata = True, output_device = device)
    
    # Calculate incoming wealth for each agent in subgraph
    graph_subset.ndata['net_trade'] = torch.zeros(agent_graph.num_nodes()).to(device)
    graph_subset.update_all(fn.u_add_v('disposable_wealth','zeros','net_trade_msg'), fn.sum('net_trade_msg', 'net_trade'))

    # Update wealth delta in agent graph
    agent_graph.ndata['net_trade'] = graph_subset.ndata['net_trade']
    agent_graph.ndata['net_trade'] = agent_graph.ndata['net_trade'] - agent_graph.ndata['disposable_wealth']

    # Conduct exchange
    agent_graph.ndata['wealth'] = agent_graph.ndata['wealth'] + agent_graph.ndata['net_trade']