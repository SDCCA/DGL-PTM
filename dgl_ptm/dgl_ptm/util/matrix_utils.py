"""This module contains functions for DGL matrix manipulations."""

import torch
import dgl


def apply_mask_to_sparse_matrix(sparse_matrix, mask):
    """
    Apply a mask to a sparse matrix and return an appropriately masked sparse matrix.

    Args:
        sparse_matrix (dgl.sparse.SparseMatrix): the original sparse matrix.
        mask (tensor): the mask to be applied.
    
    Return (dgl.sparse.SparseMatrix) the masked matrix.
    """
    return dgl.sparse.from_coo(sparse_matrix.row[mask], sparse_matrix.col[mask], sparse_matrix.val[mask], sparse_matrix.shape)

def downstream_nodes(source_id: int, destination_id: int, graph):
    """
    Collect all nodes downstream of a destination node, except the source node.

    For a bidirectional graph this is equivalent to all neighbours of the destination node.

    param:
        source_id (int): the source node ID.
        destination_id (int): the destination node ID.
        graph: dgl_ptm graph.
    return: the collection of downstream neighbors of the destination node except the source node.
    """
    successors_ = graph.successors(destination_id)
    successors = successors_[successors_ != source_id]
    return successors

def existing_connections(source_id: int, destination_ids, graph):
    """
    Identify all existing links between a source node and a list of destination nodes.

    param:
        source_id (int): the source node ID.
        destination_ids (int or iterable[int]): the destination ID(s).
    return: A tensor of bool flags where each element is True if the source and destination are connected.
    """
    existing_connection = graph.has_edges_between(source_id, destination_ids)
    return existing_connection 

def sparse_matrix_to_upper_triangular(sparse_matrix):
    """
    Select the upper triangular matrix from a sparse matrix.

    Note, any diagonal values are discarded, because we do not want self-loops.

    Args:
        sparse_matrix (dgl.sparse.SparseMatrix): the sparse matrix.
        
    Return (dgl.sparse.SparseMatrix): the upper triangular part of the sparse matrix.
    """
    mask = sparse_matrix.row < sparse_matrix.col
    return apply_mask_to_sparse_matrix(sparse_matrix, mask)

def upper_triangular_to_symmetrical(triangular):
    """
    Create a symmetrical matrix based on an upper triangular matrix.

    Note, we expect the diagonal to be zero, because we have no self-loops.

    Args:
        triangular (dgl.sparse.SparseMatrix): upper triangular matrix.
    
    Return (dgl.sparse.SparseMatrix): symmetrical matrix.
    """
    return triangular + triangular.T
