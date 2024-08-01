"""This module contains functions for computing network metrics."""

def average_degree(graph):
    """
    Compute the average degree of a graph.
    The average degree is the average number of edges per node.

    param: graph: dgl_ptm Graph
    return: float: The average degree.
    """
    return graph.number_of_edges() / graph.number_of_nodes()
