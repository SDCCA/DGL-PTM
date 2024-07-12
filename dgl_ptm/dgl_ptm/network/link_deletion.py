import dgl
import torch

def link_deletion(agent_graph, method: str, threshold: float):
    '''
        link_deletion - deletes links between agents according to a selected deletion method.

        In case of the "weighted" and "multinomial" methods, the probability of
        selecting an edge for deletion is (multiplied by) the inverse weight of that edge.
        
        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            method: deletion method. Must be either
                "probability": each edge selected idependently with equal probability,
                "weighted": each edge selected independently with weighted probability,
                "size": fixed number of edges selected with equal probability,
                "multinomial": fixed number of edges selected with weighted probability.
            threshold: Threshold for deleting an existing edge between two agent nodes.
                The interpretation of this threshold depends on the deletion method:
                "probability": the probability for deleting any edge,
                "weighted": the base probability for deleting any edge,
                "size": the number of edges to delete,
                "multinomial": the number of edges to delete.

        Output:
            agent_graph: Updated agent_graph with reduced edges based on 'method' and 'threshold'.
    '''
    agent_graph.remove_edges(_select_edges(agent_graph, method = method, threshold = threshold))


def _select_edges(agent_graph, method: str, threshold: float):
    '''
        Identify edges to delete according to a selected deletion method.

        In case of the "weighted" and "multinomial" methods, the probability of
        selecting an edge for deletion is (multiplied by) the inverse weight of that edge.
        
        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents
            method: deletion method. Must be either
                "probability": each edge selected idependently with equal probability,
                "weighted": each edge selected independently with weighted probability,
                "size": fixed number of edges selected with equal probability,
                "multinomial": fixed number of edges selected with weighted probability.
            threshold: Threshold for deleting an existing edge between two agent nodes.
                The interpretation of this threshold depends on the deletion method:
                "probability": the probability for deleting any edge,
                "weighted": the base probability for deleting any edge,
                "special": the number of edges to delete, or
                "multinomial": the number of edges to delete.

        Return:
            agent_graph.edge_ids: edge_ids for agent edges to be deleted
    '''
    upper_triangular = _sparse_upper_triangular(agent_graph.adj())

    #mask_edges
    if method == "probability":
        mask_edges = torch.rand(upper_triangular.val.size()[0]) < threshold
    elif method == "weighted":
        mask_edges = ((1.-agent_graph.edata['weight']) * torch.rand(upper_triangular.val.size()[0])) < threshold
    elif method == "size":
        mask_edges = torch.randperm(upper_triangular.val.size()[0]) < threshold
    elif method == "multinomial":
        mask_edges = torch.zeros(upper_triangular.val.size()[0])
        eid = (1.-agent_graph.edata['weight']).multinomial(threshold, replacement=False)
        mask_edges.scatter_(0, eid, 1.)
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
    Select the upper triangular matrix from a sparse matrix.

    Args:
        spm: the sparse matrix (dgl.sparse.SparseMatrix)
        
    Return: dgl.sparse.SparseMatrix
    """
    mask = spm.row < spm.col
    return _sparse_matrix_apply_mask(spm, mask)


def _symmetrical_from_upper_triangular(triu):
    """
    Create a symmetrical matrix based on an input upper triangular matrix.
    Note, this works because the diagonal is zero as we have no self-loops.

    Args:
        triu: upper triangular matrix
    
    Return: dgl.sparse.SparseMatrix
    """
    return triu + triu.T