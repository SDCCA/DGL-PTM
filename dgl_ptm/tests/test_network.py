import pytest
import dgl_ptm
import os

from dgl_ptm.network.global_attachment import global_attachment
from dgl_ptm.network.link_deletion import link_deletion
from dgl_ptm.network.local_attachment import local_attachment
from dgl_ptm.network.network_creation import network_creation


os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')
    model.set_model_parameters()
    model.initialize_model()
    return model


class TestGlobalAttachment:
    def test_global_attachment(self, model):
        ratio = model.steering_parameters['ratio']
        current_number_of_edges = model.model_graph.number_of_edges()

        global_attachment(model.model_graph, model.device, ratio)
        updated_number_of_edges = model.model_graph.number_of_edges()
        assert updated_number_of_edges > 0
        assert updated_number_of_edges > ratio * current_number_of_edges
        assert updated_number_of_edges < (1 + ratio) * current_number_of_edges


class TestLinkDeletion:
    def test_link_deletion(self, model):
        deletion_prob = model.steering_parameters['deletion_prob']
        current_number_of_edges = model.model_graph.number_of_edges()

        link_deletion(model.model_graph, deletion_prob)
        updated_number_of_edges = model.model_graph.number_of_edges()

        assert updated_number_of_edges > 0
        assert updated_number_of_edges < current_number_of_edges


class TestLocalAttachment:
    def test_local_attachment(self, model):
        current_edges = model.model_graph.number_of_edges()

        local_attachment(model.model_graph, n_FoF_links=1, edge_prop='weight', p_attach=1.)
        updated_edges = model.model_graph.number_of_edges()

        assert updated_edges >= current_edges

        nodes = model.model_graph.edges('all')[0][-2:] # new nodes

        # assert if the new nodes are in the graph
        assert nodes[0] in model.model_graph.edges('all')[1]
        assert nodes[1] in model.model_graph.edges('all')[1]


class TestNetworkCreation:
    def test_network_creation_barabasi_albert(self, model):
        agent_graph = network_creation(model.number_agents, model.initial_graph_type)

        # TODO: fix: not very informative tests due to not setting the random generator
        assert agent_graph.number_of_nodes() == model.number_agents
        assert agent_graph.number_of_edges() == model.model_graph.number_of_edges()
        assert agent_graph.number_of_nodes() == model.model_graph.number_of_nodes()

    def test_network_creation_not_implemented(self, model):
        with pytest.raises(NotImplementedError):
            network_creation(model.number_agents, 'not_implemented')
