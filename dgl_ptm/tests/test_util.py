import pytest
import dgl_ptm
import os

from dgl_ptm.util.network_metrics import mean_connectivity

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')
    model.set_model_parameters()
    model.initialize_model()
    return model


class TestNetworkMetrics:
    def test_mean_connectivity(self, model):
        mc = mean_connectivity(model.model_graph)
        assert mc is not None

        model.step() # timestep 1
        mc = mean_connectivity(model.model_graph)
        assert mc is not None
        assert isinstance(mc, float)
