import pytest
import dgl_ptm
import os

from dgl_ptm.util.network_metrics import average_degree

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='my_model', root_path='test_models')
    # to make sure the results are reproducible
    model.set_model_parameters(
        initial_graph_args={'seed': 100, 'new_node_edges': 1},
        number_agents=100,
        initial_graph_type="barabasi-albert")
    model.initialize_model()
    return model


class TestNetworkMetrics:
    def test_average_degree_initialize(self, model):
        expected_ad = 1.98
        assert model.average_degree == expected_ad

    def test_average_degree_step(self, model):
        model.step() # timestep 1
        ad = average_degree(model.model_graph)
        assert model.average_degree == ad
        assert isinstance(ad, float)
