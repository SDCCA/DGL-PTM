import pytest
import dgl_ptm
import os

from dgl_ptm.agentInteraction.trade_money import trade_money
from dgl_ptm.agentInteraction.weight_update import weight_update

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='agent_interaction', root_path='test_models')
    model.set_model_parameters(overwrite=True)
    model.initialize_model()
    return model


class TestTradeMoney:
    def test_trade_money_weighted_transfer(self, model):
        expected_disposable_wealth = model.graph.ndata['lambda'] * model.graph.ndata['wealth']

        trade_money(model.graph, 'cpu', 'weighted_transfer')

        assert (model.graph.ndata['disposable_wealth'] == expected_disposable_wealth).all()

        # assert keys are present in graph
        assert 'total_weight' in model.graph.ndata
        assert 'weight' in model.graph.edata
        assert 'disposable_wealth' in model.graph.ndata

    def test_trade_money_singular_transfer(self, model):
        expected_disposable_wealth = model.graph.ndata['lambda'] * model.graph.ndata['wealth']

        trade_money(model.graph, 'cpu', 'singular_transfer')

        assert (model.graph.ndata['disposable_wealth'] == expected_disposable_wealth).all()

        # assert keys are present in graph
        assert 'net_trade' in model.graph.ndata

    def test_trade_money_invalid_method(self, model):
        with pytest.raises(NotImplementedError):
            trade_money(model.graph, 'cpu', 'invalid_method')


class TestWeightUpdate:
    def test_weight_update(self, model):
        homophily_parameter = model.steering_parameters["homophily_parameter"]
        characteristic_distance = model.steering_parameters["characteristic_distance"]
        truncation_weight = model.steering_parameters["truncation_weight"]

        weight_update(model.graph, 'cpu', homophily_parameter, characteristic_distance, truncation_weight)

        # assert that model.graph.edata['weight'] are greater equal to truncation_weight
        assert (model.graph.edata['weight'] >= truncation_weight).all()
