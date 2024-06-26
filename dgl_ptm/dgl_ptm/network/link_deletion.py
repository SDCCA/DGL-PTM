import dgl
import torch

def link_deletion(agent_graph, del_method: str, del_threshold: float):
    '''
        link_deletion - deletes links between agents with a deletion probability 'del_prob' 
                        by sampling against a random uniform distribution.

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            del_method: deletion method; either
                "probability" (edges selected idependently by probability) or
                "size" (fixed number of randomly selected edges)
            del_threshold: Threshold for deleting an existing edge between two agent nodes.
                Either the number of edges, or the probability applied to each edge.

        Output:
            agent_graph: Updated agent_graph with reduced edges based on 'del_prob'
    '''
    agent_graph.remove_edges(_select_edges(agent_graph, del_method=del_method, del_threshold=del_threshold))


def _select_edges(agent_graph, del_method: str, del_threshold: float):
    '''
        Identify edges to delete based on a probability and triangular matrix manipulation

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            del_method: deletion method; either
                "probability" (edges selected idependently by probability) or
                "size" (fixed number of randomly selected edges)
            del_threshold: Threshold for deleting an existing edge between two agent nodes.
                Either the number of edges, or the probability applied to each edge.

        Return:
            agent_graph.edge_ids: edge_ids for agent edges to be deleted
    '''
    upper_triangular = _sparse_upper_triangular(agent_graph.adj())

    # TODO: check https://discuss.pytorch.org/t/torch-equivalent-of-numpy-random-choice/16146/16
    mask_edges
    if del_method == "probability":
        mask_edges = torch.rand(upper_triangular.val.size()[0]) < del_threshold # * triu_adj.val TODO: Is this needed?
    elif del_method == "size":
        mask_edges = torch.randperm(upper_triangular.val.size()[0]) < del_threshold # * triu_adj.val TODO: Is this needed?
    else:
        raise NotImplementedError('Currently only "probability" and "size" deletion methods are supported')
        mask_edges = torch.zeros(upper_triangular.val.size()[0])

    deletion_matrix_upper_tri = _sparse_matrix_apply_mask(upper_triangular, mask_edges)
    deletion_matrix = _symmetrical_from_upper_triangular(deletion_matrix_upper_tri)

    return agent_graph.edge_ids(deletion_matrix.row, deletion_matrix.col)


def _sparse_matrix_apply_mask(om, mask):
    """
    apply mask to a sparse matrix and return an appropriately masked sparse matrix

    Args:
        om: the original sparse matrix (dgl.sparse.SparseMatrix)
        mask: the mask to be applied (tensor)
    
    Return: dgl.sparse.SparseMatrix
    """
    return dgl.sparse.from_coo(om.row[mask], om.col[mask], om.val[mask], shape=om.shape)


def _sparse_upper_triangular(spm):
    """
    select the upper triangular matrix from a sparse matrix

    Args:
        spm: the sparse matrix (dgl.sparse.SparseMatrix)
        
    Return: dgl.sparse.SparseMatrix
    """
    mask = spm.row < spm.col
    return _sparse_matrix_apply_mask(spm, mask)


def _symmetrical_from_upper_triangular(triu):
    """
    create a symmetrical matrix based on an input upper triangular matrix. 
    Note, this works because the diagonal is zero as we have no self-loops

    Args:
        triu: upper triangular matrix
    
    Return: dgl.sparse.SparseMatrix
    """
    return triu + triu.T
