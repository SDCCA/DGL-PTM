import torch
import dgl 
from dgl.sparse import spmatrix


def local_attachment_tensor(graph,n_FoF_links,edge_prop=None,p_attach=1.):
    '''This function attempts to form links between two agents connected by a common neighbor
    regardless of weight. Notes: n_FoF_links represents attempted links; if a connecting 
    agent has insufficient neighbors or the neighbors are already connected or
    the random number generated is less than p_attach, a new link is not formed.'''
    # Select bridge/connecting nodes
    connecting_nodes = torch.randint(0,graph.nodes().size(0),(n_FoF_links,))

    # Sample 2 neighbors of bridge/connecting nodes
    sample = dgl.sampling.sample_neighbors(graph,connecting_nodes, 2 , edge_dir="out")

    # Record connecting node degree
    sample.ndata['out_degree'] = sample.out_degrees()

    # Identify edges with no counterpart (i.e. Cases of connecting nodes with <2 neighbors)
    def edges_to_remove(edges):
        return (edges.src['out_degree'] != 2)

    insufficient_connections = sample.filter_edges(edges_to_remove)

    # Remove offenders
    sample.remove_edges(insufficient_connections)

    # Extract node pairs and exclude existing edges
    existing_connections = graph.has_edges_between(sample.edges(order='eid')[1][::2], sample.edges(order='eid')[1][1::2])
    even_indices_tensor = sample.edges(order='eid')[1][::2][~existing_connections]
    odd_indices_tensor = sample.edges(order='eid')[1][1::2][~existing_connections]  

    # Compare random number for each prospective link to p_attach
    successful_links = torch.rand(even_indices_tensor.size(0)) < p_attach

    # Add new edges to the original graph
    graph.add_edges(even_indices_tensor[successful_links], odd_indices_tensor[successful_links])
    graph.add_edges(odd_indices_tensor[successful_links], even_indices_tensor[successful_links])


