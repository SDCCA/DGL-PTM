# import os
# import pytest
# import dgl_ptm
# import torch
# from dgl_ptm.agent.capital_update import capital_update
# from dgl_ptm.agent.income_generation import income_generation
# from dgl_ptm.agent.wealth_consumption import wealth_consumption
# from dgl_ptm.agent.agent_update import agent_update

# os.environ["DGLBACKEND"] = "pytorch"

# @pytest.fixture
# def model():
#     model = dgl_ptm.PovertyTrapModel(model_identifier='my_model')
#     model.set_model_parameters()
#     model.initialize_model()
#     return model

# class TestCapitalUpdate:

#     def test_capital_update_default(self, model):

#         k = model.model_graph.ndata['wealth']
#         c = model.model_graph.ndata['wealth_consumption']
#         i_a = model.model_graph.ndata['i_a']
#         m = model.model_graph.ndata['m']
#         global_θ =model.model_data['modelTheta'][0]
#         𝛿 = model.steering_parameters['depreciation']
#         # generate random income with size 100
#         income = torch.rand(100)
#         expected_wealth = (global_θ + m * (1-global_θ)) * (income - c - i_a + (1-𝛿) * k)

#         # because income_generation is not called yet
#         model.model_graph.ndata['income'] = income

#         capital_update(
#             model.model_graph,
#             model.steering_parameters,
#             model.model_data,
#             1,
#             method="default"
#             )

#         # assert _agent_capital_update was called
#         assert (model.model_graph.ndata['wealth'] == expected_wealth).all()

#     def test_capital_update_other(self, model):
#         # assert NotImplementedError was raised
#         with pytest.raises(NotImplementedError):
#             capital_update(
#                 model.model_graph,
#                 model.model_data,
#                 1,
#                 method='other'
#                 )


# class TestIncomeGeneration:

#     def test_income_generation_default(self, model):
#         gamma = model.steering_parameters['tech_gamma']
#         cost = model.steering_parameters['tech_cost']
#         alpha = model.model_graph.ndata['alpha']
#         wealth = model.model_graph.ndata['wealth']
#         expected_income,expected_tech_index = torch.max((alpha[:,None]*wealth[:,None]**gamma - cost), axis=1)

#         income_generation(
#             model.model_graph,
#             model.device,
#             model.steering_parameters,
#             method='income_generation'
#             )

#         # assert _agent_income_generator was called
#         assert (model.model_graph.ndata['income'] == expected_income).all()
#         assert (model.model_graph.ndata['tech_index'] == expected_tech_index).all()

#     def test_income_generation_other(self, model):
#         # assert NotImplementedError was raised
#         with pytest.raises(NotImplementedError):
#             income_generation(
#                 model.model_graph,
#                 model.device,
#                 model.steering_parameters,
#                 method='other'
#                 )


# class TestWealthConsumption:

#     def test_wealth_consumption_default(self, model):
#         wealth = model.model_graph.ndata['wealth']
#         expected_wealth_consumption = 0.64036047*torch.log(wealth)

#         wealth_consumption(
#             model.model_graph,
#             model.steering_parameters,
#             method='fitted_consumption'
#             )

#         # assert _fitted_agent_wealth_consumption was called
#         assert (model.model_graph.ndata['wealth_consumption'] == expected_wealth_consumption).all()

#     def test_wealth_consumption_other(self, model):
#         # assert NotImplementedError was raised
#         with pytest.raises(NotImplementedError):
#             wealth_consumption(
#                 model.model_graph,
#                 model.steering_parameters,
#                 method='other'
#                 )

#     def test_wealth_consumption_bellman(self, model):
#         # assert NotImplementedError was raised
#         with pytest.raises(NotImplementedError):
#             wealth_consumption(
#                 model.model_graph,
#                 model.steering_parameters,
#                 method='bellman_consumption'
#                 )


# class TestAgentUpdate:
#     def test_agent_update_default(self, model):
#         # assert NotImplementedError was raised
#         with pytest.raises(NotImplementedError):
#             agent_update(
#                 model.model_graph,
#                 model.device,
#                 model.steering_parameters,
#                 model.model_data,
#                 1,
#                 method='default'
#                 )

#     def test_agent_update_capital(self, model):
#         # because income_generation is not called yet
#         model.model_graph.ndata['income'] = torch.rand(100)
#         agent_update(
#             model.model_graph,
#             model.device,
#             model.steering_parameters,
#             model.model_data,
#             1,
#             method='capital'
#             )
#         assert (model.model_graph.ndata['wealth'] != 0).all()

#     def test_agent_update_theta(self, model):
#         agent_update(
#             model.model_graph,
#             model.device,
#             model.steering_parameters,
#             model.model_data,
#             1,
#             method='theta'
#             )
#         assert (model.model_graph.ndata['theta'] != 0).all()

#     def test_agent_update_consumption(self, model):
#         agent_update(
#             model.model_graph,
#             model.device,
#             model.steering_parameters,
#             model.model_data,
#             1,
#             method='consumption'
#             )
#         assert (model.model_graph.ndata['wealth_consumption'] != 0).all()

#     def test_agent_update_income(self, model):
#         agent_update(
#             model.model_graph,
#             model.device,
#             model.steering_parameters,
#             model.model_data,
#             1,
#             method='income'
#             )

#         assert (model.model_graph.ndata['income'] != 0).all()
#         assert (model.model_graph.ndata['tech_index'] == 0).all() # all indices are 0
