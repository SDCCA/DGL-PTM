import pytest
import dgl_ptm
import os

from dgl_ptm.util.network_metrics import mean_connectivity
from dgl_ptm.util.network_metrics import degree_distribution
from dgl_ptm.util.network_metrics import modal_degree
from dgl_ptm.util.network_metrics import skewness_of_degree_distribution

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

    def test_degree_distribution(self, model):
        dd = degree_distribution(model.model_graph)
        assert dd is not None

        model.step() # timestep 1
        dd = degree_distribution(model.model_graph)
        assert dd is not None
        assert isinstance(dd, dict)


    def test_modal_degree(self, model):
        md = modal_degree(model.model_graph)
        assert md is not None

        model.step()
        md = modal_degree(model.model_graph)
        assert md is not None
        assert isinstance(md, int)

    def test_skewness_of_degree_distribution(self, model):
        sd = skewness_of_degree_distribution(model.model_graph)
        assert sd is not None

        model.step()
        sd = skewness_of_degree_distribution(model.model_graph)
        assert sd is not None
        assert isinstance(sd, float)
