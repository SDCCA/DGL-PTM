import pytest
import dgl_ptm
import os
import xarray as xr

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
        data_collection(model.model_graph, timestep=0, npath = model.steering_parameters['npath'],
                        epath = model.steering_parameters['epath'], ndata = model.steering_parameters['ndata'],
                        edata = model.steering_parameters['edata'], format = model.steering_parameters['format'],
                        mode = model.steering_parameters['mode'])

        assert Path('my_model/agent_data.zarr').exists()
        assert Path('my_model/edge_data/0.zarr').exists()

    def test_data_collection_time_step1(self, model):
        model.step() # timestep 1
        data_collection(model.model_graph, timestep=1, npath = model.steering_parameters['npath'],
                        epath = model.steering_parameters['epath'], ndata = model.steering_parameters['ndata'],
                        edata = model.steering_parameters['edata'], format = model.steering_parameters['format'],
                        mode = model.steering_parameters['mode'])

        assert Path('my_model/agent_data.zarr').exists()
        assert Path('my_model/edge_data/0.zarr').exists()
        assert Path('my_model/edge_data/1.zarr').exists()

        # check if dimension 'n_time' exist in agent_data.zarr
        agent_data = xr.open_zarr('my_model/agent_data.zarr')
        assert 'n_time' in agent_data.dims

        # check variable names in edge_data/1.zarr
        edge_data = xr.open_zarr('my_model/edge_data/1.zarr')
        assert 'weight' in edge_data.variables


class TestInitializeModel:
    def test_set_model_parameters(self):
        model = dgl_ptm.PovertyTrapModel(model_identifier='test_model')
        model.set_model_parameters()

        assert model.steering_parameters['npath'] == Path('test_model/agent_data.zarr')
        assert model.steering_parameters['epath'] == Path('test_model/edge_data')
        assert Path('test_model/test_model.yaml').exists()
