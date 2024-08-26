"""This module contains functions for DGL matrix manipulations."""

import torch
import dgl


def apply_mask_to_sparse_matrix(sparse_matrix, mask):
    """
    Apply a mask to a sparse matrix and return an appropriately masked sparse matrix.

    Args:
        sparse_matrix: dgl.sparse.SparseMatrix, the original sparse matrix.
        mask: tensor, the mask to be applied.
    
    Returns:
        dgl.sparse.SparseMatrix, the masked matrix.
    """
    return dgl.sparse.from_coo(sparse_matrix.row[mask], sparse_matrix.col[mask], sparse_matrix.val[mask], sparse_matrix.shape)

def downstream_nodes(source_id: int, destination_id: int, graph):
    """
    Collect all nodes downstream of a destination node, except the source node.

    For a bidirectional graph this is equivalent to all neighbours of the destination node.

    Args:
        source_id: int, the source node ID.
        destination_id: int, the destination node ID.
        graph: dgl_ptm graph.
    Returns:
        the collection of downstream neighbors of the destination node except the source node.
    """
    successors_ = graph.successors(destination_id)
    successors = successors_[successors_ != source_id]
    return successors

def existing_connections(source_id: int, destination_ids, graph):
    """
    Identify all existing links between a source node and a list of destination nodes.

    Args:
        source_id: int, the source node ID.
        destination_ids: int or iterable[int], the destination ID(s).
    Returns:
        A tensor of bool flags where each element is True if the source and destination are connected.
    """
    existing_connection = graph.has_edges_between(source_id, destination_ids)
    return existing_connection 

def sparse_matrix_to_triangular(sparse_matrix,selection=None):
    """
    Select either the upper or lower triangular matrix
    from a sparse matrix.

    Note, diagonal elements are discarded to avoid self-loops.

    Args:
        sparse_matrix: dgl.sparse.SparseMatrix, the sparse matrix.
        selection: string, `upper` or `lower` specifying which form.

    Returns:
        dgl.sparse.SparseMatrix, the specified triangular part of the sparse matrix.
    """
    if selection == 'upper':
        mask = sparse_matrix.row < sparse_matrix.col
    elif selection == 'lower':
        mask = sparse_matrix.row > sparse_matrix.col
    else:
        raise ValueError('triangular matrix not specified. Must be upper or lower')
    return apply_mask_to_sparse_matrix(sparse_matrix, mask)

def sparse_matrix_to_upper_triangular(sparse_matrix):
    """
    Select the upper triangular matrix from a sparse matrix.

    Note, any diagonal values are discarded, because we do not want self-loops.

    Args:
        sparse_matrix: dgl.sparse.SparseMatrix, the sparse matrix.
        
    Returns:
        dgl.sparse.SparseMatrix, the upper triangular part of the sparse matrix.
    """
    return sparse_matrix_to_triangular(sparse_matrix, selection='upper')

def sparse_matrix_to_lower_triangular(sparse_matrix):
    """
    Select the lower triangular matrix from a sparse matrix.

    Note, any diagonal values are discarded, because we do not want self-loops.

    Args:
        sparse_matrix: dgl.sparse.SparseMatrix, the sparse matrix.
        
    Returns:
        dgl.sparse.SparseMatrix, the lower triangular part of the sparse matrix.
    """
    return sparse_matrix_to_triangular(sparse_matrix, selection='lower')

def upper_triangular_to_symmetrical(triangular):
    """
    Create a symmetrical matrix based on an upper triangular matrix.

    Note, we expect the diagonal to be zero, because we have no self-loops.

    Args:
        triangular: dgl.sparse.SparseMatrix, upper triangular matrix.
    
    Returns:
        dgl.sparse.SparseMatrix, symmetrical matrix.
    """
    return triangular + triangular.T

def clear_symmetrical_duplicates(sparse_matrix):
    """
    Remove possible duplicate edges.

    Assuming a matrix (nxn) with non-zero elements spread throughout both upper
    lower triangular sub-matrices is a sub-selectio of a symmetrical matrix, 
    remove duplication of link.
    Note: This assumes and no diagonal elements (i.e self-loops)
    are present.

    Args:
        sparse_matrix: dgl.sparse.SparseMatrix, the sparse matrix.
    Returns:
        deduplicated_matrix: dgl.sparse.SparseMatrix, the de-duplicated matrix
        duplicate_ones: dgl.sparse.SparseMatrix, sparse matrix with duplicate elements 
        set to 1    
    """
    upper_triangular = sparse_matrix_to_upper_triangular(sparse_matrix)
    lower_triangular = sparse_matrix_to_lower_triangular(sparse_matrix)
    duplicate_links_prod = upper_triangular.T*lower_triangular
    duplicate_links_mask = duplicate_links_prod.val > 0
    duplicate_links = apply_mask_to_sparse_matrix(duplicate_links_prod, duplicate_links_mask)
    duplicate_ones = dgl.sparse.val_like(duplicate_links,torch.ones(duplicate_links.nnz, dtype=int))
    deduplicated_matrix = sparse_matrix - sparse_matrix*duplicate_ones
    return deduplicated_matrix, duplicate_ones