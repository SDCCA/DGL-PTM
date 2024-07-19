"""This module contains functions for computing network metrics."""

def mean_connectivity(graph):
    """
    Compute the mean connectivity of a graph.
    The mean connectivity is the average number of edges per node.

    param: graph: dgl_ptm Graph
    return: float: The mean connectivity.
    """
    return graph.number_of_edges() / graph.number_of_nodes()
