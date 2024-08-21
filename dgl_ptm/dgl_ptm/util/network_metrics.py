"""This module contains functions for computing network metrics."""

import torch

def average_degree(graph):
    """
    Compute the average degree of a graph.
    The average degree is the average number of edges per node.

    param: graph: dgl_ptm Graph
    return: float: The average degree.
    """
    return graph.number_of_edges() / graph.number_of_nodes()


def average_weighted_degree(graph):
    """
    Compute the average weighted degree of a graph.
    The average weighted degree is the sum of the weights of 
    all edges divided by the number of nodes.

    param: graph: dgl_ptm Graph
    return: float: The average weighted degree.
    """
    return torch.sum(graph.edata["weight"]) / graph.number_of_nodes()