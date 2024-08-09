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
    model = dgl_ptm.PovertyTrapModel(model_identifier='network', root_path='test_models')
    model.set_model_parameters()
    model.initialize_model()
    return model


class TestGlobalAttachment:
    def test_global_attachment(self, model):
        ratio = model.steering_parameters['noise_ratio']
        current_number_of_edges = model.graph.number_of_edges()

        global_attachment(model.graph, model.config.device, ratio)
        updated_number_of_edges = model.graph.number_of_edges()
        assert updated_number_of_edges > 0
        assert updated_number_of_edges > ratio * current_number_of_edges
        assert updated_number_of_edges < (1 + ratio) * current_number_of_edges


class TestLinkDeletion:
    def test_link_deletion(self, model):
        del_method = model.steering_parameters['del_method']
        del_threshold = model.steering_parameters['del_threshold']
        current_number_of_edges = model.graph.number_of_edges()

        link_deletion(model.graph, method = del_method, threshold = del_threshold)
        updated_number_of_edges = model.graph.number_of_edges()

        assert updated_number_of_edges > 0
        assert updated_number_of_edges < current_number_of_edges


class TestLocalAttachment:
    def test_local_attachment(self, model):
        current_edges = model.graph.number_of_edges()

        local_attachment(model.graph, n_FoF_links=1, edge_prop='weight', p_attach=1.)
        updated_edges = model.graph.number_of_edges()

        assert updated_edges >= current_edges

        nodes = model.graph.edges('all')[0][-2:] # new nodes

        # assert if the new nodes are in the graph
        assert nodes[0] in model.graph.edges('all')[1]
        assert nodes[1] in model.graph.edges('all')[1]


class TestNetworkCreation:
    def test_network_creation_barabasi_albert(self, model):
        agent_graph = network_creation(model.config.number_agents, model.config.initial_graph_type)

        # TODO: fix: not very informative tests due to not setting the random generator
        assert agent_graph.number_of_nodes() == model.config.number_agents
        assert agent_graph.number_of_edges() == model.graph.number_of_edges()
        assert agent_graph.number_of_nodes() == model.graph.number_of_nodes()

    def test_network_creation_not_implemented(self, model):
        with pytest.raises(NotImplementedError):
            network_creation(model.config.number_agents, 'not_implemented')
