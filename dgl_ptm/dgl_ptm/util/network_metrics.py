"""This module contains functions for computing network metrics."""

from scipy import stats
from collections import Counter

def mean_connectivity(graph):
    """
    Compute the mean connectivity of a graph.
    The mean connectivity is the average number of edges per node.

    param: graph: dgl_ptm Graph
    return: float: The mean connectivity.
    """
    return graph.number_of_edges() / graph.number_of_nodes()


def degree_distribution(graph):
    """
    Compute the degree distribution of a graph.
    The degree distribution is a probability distribution of the degrees over
    all nodes in the network.

    param: graph: dgl_ptm Graph
    return: dict: The degree distribution.
    """
    graph = graph.to_networkx()
    degrees = [degree for node, degree in graph.degree()]
    degree_counts = Counter(degrees)
    total_nodes = len(degrees)
    distribution = {
        degree: count / total_nodes for degree, count in degree_counts.items()
        }
    return distribution


def modal_degree(graph):
    """
    Compute the modal degree of a graph.
    It finds the degree that appears most frequently in the degree distribution.

    param: graph: dgl_ptm Graph
    return: int: The modal degree.
    """
    distribution = degree_distribution(graph)
    return max(distribution, key=distribution.get)


def skewness_of_degree_distribution(graph):
    """
    Compute the skewness of the degree distribution of a graph.

    param: graph: dgl_ptm Graph
    return: float: The skewness of the degree distribution.
    """
    graph = graph.to_networkx()
    degrees = [degree for node, degree in graph.degree()]
    return stats.skew(degrees)
