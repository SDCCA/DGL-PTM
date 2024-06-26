import pytest
import dgl_ptm
import os

from dgl_ptm.model.data_collection import data_collection
from pathlib import Path

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')
    model.set_model_parameters()
    model.initialize_model()
    return model


class TestPtmStep:
    def test_ptm_step_timestep1(self, model):
        model.step() # timestep 1

        assert 'disposable_wealth' in model.model_graph.ndata
        assert 'theta' in model.model_graph.ndata
        assert 'wealth_consumption' in model.model_graph.ndata
        assert 'income' in model.model_graph.ndata

        assert Path('./my_model/agent_data.zarr').exists()
        assert Path('./my_model/edge_data/1.zarr').exists()

    def test_ptm_step_timestep2(self, model):
        model.step() # timestep 1
        k,c,i_a,m = model.model_graph.ndata['wealth'],model.model_graph.ndata['wealth_consumption'],model.model_graph.ndata['i_a'],model.model_graph.ndata['m']
        global_θ =model.model_data['modelTheta'][1]
        𝛿=model.steering_parameters['depreciation']
        new_wealth = (global_θ + m * (1-global_θ)) * (model.model_graph.ndata['income'] - c - i_a + (1-𝛿) * k)


        model.step() # timestep 2
        assert (model.model_graph.ndata['wealth'] == new_wealth).all()
        assert Path('./my_model/edge_data/2.zarr').exists()


class TestDataCollection:
    def test_data_collection(self, model):

        data_collection(model.model_graph, 2, npath='./my_model/agent_data', epath='./my_model/edge_data', ndata = ['all'], edata = ['all'], format = 'xarray', mode = 'w-')

        assert Path('./my_model/agent_data/agent_data.zarr').exists()
        assert Path('./my_model/edge_data/2.zarr').exists()
