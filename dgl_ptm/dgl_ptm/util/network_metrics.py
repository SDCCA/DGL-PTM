"""This module contains functions for computing network metrics."""

import torch
import dgl.sparse as sparse

def average_degree(graph):
    """
    Compute the average degree of a graph.
    The average degree is the average number of edges per node.
    Note that the result is doubled for in and out combined.

    param: graph: dgl_ptm graph
    return: float: The average degree.
    """
    return 2 * graph.number_of_edges() / graph.number_of_nodes()


def average_weighted_degree(graph):
    """
    Compute the average weighted degree of a graph.
    The average weighted degree is the sum of the weights of 
    all edges divided by the number of nodes.
    Note that the result is doubled for in and out combined.

    param: graph: dgl_ptm graph
    return: float: The average weighted degree.
    """
    return 2 * torch.sum(graph.edata["weight"]) / graph.number_of_nodes()

def node_degree(graph):
    """
    Compute the degree of each node in the graph.
    Note that the result is in and out combined.

    param: graph: dgl_ptm graph
    return: torch.Tensor: The degree of each node.
    """
    return graph.in_degrees() + graph.out_degrees()

def node_weighted_degree(graph):
    """
    Compute the weighted degree of each node in the graph.
    Note that the result is in and out combined.

    param: graph: dgl_ptm graph
    return: torch.Tensor: The weighted degree of each node.
    """
    weights_sparse = sparse.spmatrix(torch.stack(graph.edges(order='eid'),dim=0), graph.edata['weight'],shape=(graph.number_of_nodes(),graph.number_of_nodes()))

    return sparse.sum(weights_sparse, dim=0) + sparse.sum(weights_sparse, dim=1)