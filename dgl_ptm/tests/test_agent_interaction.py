import pytest
import dgl_ptm
import os

from dgl_ptm.agentInteraction.trade_money import trade_money
from dgl_ptm.agentInteraction.weight_update import weight_update

os.environ["DGLBACKEND"] = "pytorch"

@pytest.fixture
def model():
    model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')
    model.set_model_parameters()
    model.initialize_model()
    return model


class TestTradeMoney:
    def test_trade_money_weighted_transfer(self, model):
        trade_money(model.model_graph, 'cpu', 'weighted_transfer')

        expected_disposable_wealth = model.model_graph.ndata['lambda'] * model.model_graph.ndata['wealth']
        assert (model.model_graph.ndata['disposable_wealth'] == expected_disposable_wealth).all()

        # assert keys are present in model_graph
        assert 'total_weight' in model.model_graph.ndata
        assert 'weight' in model.model_graph.edata
        assert 'disposable_wealth' in model.model_graph.ndata

    def test_trade_money_singular_transfer(self, model):
        trade_money(model.model_graph, 'cpu', 'singular_transfer')

        expected_disposable_wealth = model.model_graph.ndata['lambda'] * model.model_graph.ndata['wealth']
        assert (model.model_graph.ndata['disposable_wealth'] == expected_disposable_wealth).all()

        # assert keys are present in model_graph
        assert 'net_trade' in model.model_graph.ndata

    def test_trade_money_invalid_method(self, model):
        with pytest.raises(NotImplementedError):
            trade_money(model.model_graph, 'cpu', 'invalid_method')


class TestWeightUpdate:
    def test_weight_update(self, model):
        homophily_parameter = model.steering_parameters["homophily_parameter"]
        characteristic_distance = model.steering_parameters["characteristic_distance"]
        truncation_weight = model.steering_parameters["truncation_weight"]

        weight_update(model.model_graph, 'cpu', homophily_parameter, characteristic_distance, truncation_weight)

        # assert that model.model_graph.edata['weight'] are greater equal to truncation_weight
        assert (model.model_graph.edata['weight'] >= truncation_weight).all()
