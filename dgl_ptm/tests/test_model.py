import pytest
import dgl_ptm
import os
import xarray as xr
import torch

from dgl_ptm.model.data_collection import data_collection
from dgl_ptm.model.initialize_model import sample_distribution_tensor
from pathlib import Path

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')
    model.set_model_parameters()
    model.initialize_model()
    return model


@pytest.fixture
def config_file(tmp_path):
    with open(tmp_path / 'test_config.yaml', 'w') as file:
        file.write('''
        model_identifier: new_model
        steering_parameters:
            deletion_prob: 0.04
        ''')
    return tmp_path / 'test_config.yaml'


class TestPtmStep:
    def test_ptm_step_timestep0(self, model):
        model.step() # timestep 0

        assert 'disposable_wealth' in model.model_graph.ndata
        assert 'theta' in model.model_graph.ndata
        assert 'wealth_consumption' in model.model_graph.ndata
        assert 'income' in model.model_graph.ndata

        assert Path('my_model/agent_data.zarr').exists()
        assert Path('my_model/edge_data/0.zarr').exists()

    def test_ptm_step_timestep1(self, model):
         model.step() # timestep 0
         model.step() # timestep 1
         assert Path('my_model/edge_data/1.zarr').exists()


class TestDataCollection:
    def test_data_collection(self, model):
        data_collection(model.model_graph, timestep=0, npath = model.steering_parameters['npath'],
                        epath = model.steering_parameters['epath'], ndata = model.steering_parameters['ndata'],
                        edata = model.steering_parameters['edata'], format = model.steering_parameters['format'],
                        mode = model.steering_parameters['mode'])

        assert Path('my_model/agent_data.zarr').exists()
        assert Path('my_model/edge_data/0.zarr').exists()

    def test_data_collection_timestep1(self, model):
        model.step() # timestep 0
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

        assert Path(model.steering_parameters['npath']) == Path('test_model/agent_data.zarr')
        assert Path(model.steering_parameters['epath']) == Path('test_model/edge_data')
        assert Path('test_model/test_model.yaml').exists()
        assert model.steering_parameters['edata'] == ['all']
        assert model.steering_parameters['format'] == 'xarray'
        assert model.number_agents == 100
        assert model.step_target == 5

    def test_set_model_parameters_with_file(self, config_file):
        model = dgl_ptm.PovertyTrapModel(model_identifier='test_model')
        model.set_model_parameters(parameterFilePath=config_file)

        assert model._model_identifier == 'test_model'
        assert model.steering_parameters['del_method'] == 'probability'
        assert model.steering_parameters['del_threshold'] == 0.05
        assert model.number_agents == 100

    def test_set_model_parameters_with_kwargs(self):
        model = dgl_ptm.PovertyTrapModel(model_identifier='test_model')
        model.set_model_parameters(steering_parameters={'del_method': 'probability','del_threshold': 0.04})

        assert model.steering_parameters['del_method'] == 'probability'
        assert model.steering_parameters['del_threshold'] == 0.04
        assert model.number_agents == 100

    def test_set_model_parameters_with_file_and_kwargs(self, config_file):
        model = dgl_ptm.PovertyTrapModel(model_identifier='test_model')
        model.set_model_parameters(
            parameterFilePath=config_file,
            steering_parameters={'del_method': 'probability','del_threshold': 0.06}
            )

        assert model.steering_parameters['del_method'] == 'probability'
        assert model.steering_parameters['del_threshold'] == 0.06
        assert model.number_agents == 100

    def test_initialize_model(self, model):
        assert model.model_graph is not None
        assert model.number_of_edges is not None
        assert Path('my_model/edge_data/0.zarr').exists()
        assert str(model.model_graph.device) == 'cpu'

    def test_create_network(self):
        model = dgl_ptm.PovertyTrapModel(model_identifier='test_model')
        model.set_model_parameters()
        model.create_network()

        assert model.model_graph is not None
        assert model.model_graph.number_of_nodes() == 100

    def test_initialize_model_properties(self):
        model = dgl_ptm.PovertyTrapModel(model_identifier='test_model')
        model.set_model_parameters()
        model.initialize_model_properties()

        modelTheta = torch.tensor([1., 1., 1., 1., 1.])

    def test_initialize_agent_properties(self):
        model = dgl_ptm.PovertyTrapModel(model_identifier='test_model')
        model.set_model_parameters()
        model.create_network()
        model.initialize_agent_properties()

        wealth_consumption = torch.zeros(model.model_graph.num_nodes())
        ones = torch.ones(model.model_graph.num_nodes())

        assert model.model_graph.ndata['wealth'] is not None
        assert model.model_graph.ndata['alpha'] is not None
        assert model.model_graph.ndata['theta'] is not None
        assert model.model_graph.ndata['sensitivity'] is not None
        assert model.model_graph.ndata['lambda'] is not None
        assert (model.model_graph.ndata['wealth_consumption'] == wealth_consumption).all()
        assert (model.model_graph.ndata['ones'] == ones).all()

    def test_step(self, model):
        model.step()

        assert model.step_count == 1
        assert model.model_graph.number_of_nodes() == 100

    def test_run(self, model):
        model.run()

        assert model.step_count == 5
        assert model.model_graph.number_of_nodes() == 100

    def test_model_init_savestate(self, model):
        model.checkpoint_period = 1
        model.run()

        assert model.inputs is not None
        assert Path('my_model/model_graph.bin').exists()
        assert Path('my_model/generator_state.bin').exists()
        assert model.inputs["step_count"] == 5

    def test_model_init_savestate_not_default(self, model):
        model.checkpoint_period = 2
        model.run()

        assert model.inputs["step_count"] == 4

    def test_model_init_restart(self, model):
        model.checkpoint_period = 1
        model.step_target = 3 # only run the model till step 3
        model.run()
        expected_generator_state = set(model.inputs["generator_state"].tolist())

        model.restart = True
        model.step_target = 5 # contiune the model till step 5
        model.run()
        stored_generator_state = set(model.inputs["generator_state"].tolist())

        assert model.inputs is not None
        assert model.inputs["step_count"] == 5
        assert stored_generator_state == expected_generator_state 
