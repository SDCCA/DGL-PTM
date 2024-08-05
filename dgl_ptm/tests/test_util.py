import pytest
import dgl_ptm
import os

from dgl_ptm.util.network_metrics import average_degree

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')
    model.set_model_parameters()
    model.initialize_model()
    return model


class TestNetworkMetrics:
    def test_average_degree_initialize(self, model):
        ad = average_degree(model.model_graph)
        assert model.average_degree == ad
        assert isinstance(ad, float)

    def test_average_degree_step(self, model):
        model.step() # timestep 1
        ad = average_degree(model.model_graph)
        assert model.average_degree == ad
        assert isinstance(ad, float)
