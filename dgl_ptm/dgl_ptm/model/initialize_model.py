import copy
from logging import Logger
from pathlib import Path
import dgl
import torch
import yaml
import pickle

from dgl_ptm.network.network_creation import network_creation
from dgl_ptm.model.step import ptm_step
from dgl_ptm.agentInteraction.weight_update import weight_update
from dgl_ptm.model.data_collection import data_collection
from dgl.data.utils import save_graphs, load_graphs

logger = Logger(__name__)

# Set the seed of the random number generator
# this is global and will affect all random number generators
generator = torch.manual_seed(0)


def sample_distribution_tensor(type, distParameters, nSamples, round=False, decimals=None):
    """
    create and return samples from different distributions

    :param type: Type of distribution to sample
    :param distParameters: array of parameters as required/supported by requested distribution type
    :param nSamples: number of samples to return (as 1d tensor)
    :param round: optional, whether the samples are to be rounded
    :param decimals: optional, required if round is specified. decimal places to round to
    """

    if type == 'uniform':
        dist = torch.distributions.uniform.Uniform(torch.tensor(distParameters[0]),torch.tensor(distParameters[1])).sample(torch.tensor([nSamples]))
    elif type == 'normal':
        dist = torch.distributions.normal.Normal(torch.tensor(distParameters[0]),torch.tensor(distParameters[1])).sample(torch.tensor([nSamples]))
    elif type == 'bernoulli':
        dist = torch.distributions.bernoulli.Bernoulli(probs=distParameters[0],logits=distParameters[1],validate_args=None).sample(torch.tensor([nSamples]))
    elif type == 'multinomial':
        dist = torch.gather(torch.Tensor(distParameters[1]), 0, torch.multinomial(torch.tensor(distParameters[0]), nSamples, replacement=True))

    else:
        raise NotImplementedError('Currently only uniform, normal, multinomial, and bernoulli distributions are supported')

    if round:
        if decimals == None:
            raise ValueError('rounding requires decimals of rounding accuracy to be specified')
        else:
            return torch.round(dist,decimals=decimals)
    else:
        return dist

class Model(object):
    """
    Abstract model class
    """

    def __init__(self,model_identifier=None):
        self._model_identifier = model_identifier
        self.number_agents = None

    def create_network(self):
        raise NotImplementedError('network creaion is not implemented for this class.')

    def step(self):
        raise NotImplementedError('step function is not implemented for this class.')

    def run(self):
        raise NotImplementedError('run method is not implemented for this class.')

class PovertyTrapModel(Model):
    """
    Poverty Trap model as derived model class

    """

    #default values as class variable
    default_model_parameters = {'number_agents': 100 ,
    'gamma_vals':torch.tensor([0.3,0.45]) , #for pseudo income
    'sigma_dist': {'type':'uniform','parameters':[0.1,1.9],'round':True,'decimals':1},
    'cost_vals': torch.tensor([0.,0.45]), #for pseudo income
    'technology_levels': torch.tensor([0,1]), #check if deletable
    'a_theta_dist': {'type':'uniform','parameters':[0.1,1],'round':False,'decimals':None},
    'sensitivity_dist':{'type':'uniform','parameters':[0.0,1],'round':False,'decimals':None},
    'technology_dist': {'type':'bernoulli','parameters':[0.5,None],'round':False,'decimals':None},
    'capital_dist': {'type':'uniform','parameters':[0.1,10.],'round':False,'decimals':None},
    'alpha_dist': {'type':'normal','parameters':[1.08,0.074],'round':False,'decimals':None},
    'lambda_dist': {'type':'uniform','parameters':[0.1,0.9],'round':True,'decimals':1},
    'initial_graph_type': 'barabasi-albert',
    'step_count':0,
    'step_target':20,
    'model_data':{},
    'steering_parameters':{'npath':'./agent_data.zarr',
                            'epath':'./edge_data',
                            'ndata':['all_except',['a_table']],
                            'edata':['all'],
                            'format':'xarray',
                            'mode':'w-',
                            'wealth_method':'singular_transfer',
                            'income_method':'default',
                            'capital_update_method':'default',
                            'consume_method':'default',
                            'perception_method':'default',
                            'tech_gamma': torch.tensor([0.3,0.35,0.45]),
                            'tech_cost': torch.tensor([0,0.15,0.65]),
                            'adapt_m':torch.tensor([0,0.5,0.9]),
                            'adapt_cost':torch.tensor([0,0.25,0.45]),
                            'depreciation': 0.6,
                            'discount': 0.95,
                            'm_theta_dist': {'type':'multinomial','parameters':[[0.02 ,0.03, 0.05, 0.9],[0.7, 0.8, 0.9, 1]],'round':False,'decimals':None},
                            'deletion_prob':0.05,
                            'ratio':0.1,
                            'homophily_parameter':0.69,
                            'characteristic_distance':35,
                            'truncation_weight':1.0e-10,}
    }

    def __init__(self,*, model_identifier, restart=None, savestate=True):
        """
        restore from a savestate or create a PVT model instance.
        Checks whether a model indentifier has been specified.

        param: model_identifier: str, required. Identifier for the model. Used to save and load model states.
        param: restart: int, optional. If specified, the model is run from
        this step. Default None.
        param: savestate: bool, optional. If True, the states are saved to files. Default True.
        """
        super().__init__(model_identifier = model_identifier)

        self.restart = restart
        if self.restart == 0:
            raise ValueError('restart step should be greater than 0. '
                             'If you want to start from the beginning, '
                             'do not specify restart step and run intialize_model()')


        if self.restart:
            path_model_graph = f'./{self._model_identifier}/model_graphs.bin'
            path_generator_state = f'./{self._model_identifier}/generator_state.bin'

            if Path(path_model_graph).is_file() and Path(path_generator_state).is_file():
                try:
                    self.inputs = _load_model(f'./{self._model_identifier}')
                    logger.info(f'Some model logs found, model {self._model_identifier} loaded')
                except:
                    raise ValueError(f'Loading model from files {path_model_graph} and {path_generator_state} failed')

            self.step_count = self.restart - 1
            if not self.inputs.get(self.step_count):
                raise ValueError(f'No inputs for previous step {self.step_count} found')

        self.savestate = savestate

        self.number_agents = None
        self.gamma_vals = None
        self.sigma_dist = None
        self.cost_vals = None
        self.technology_levels = None
        self.technology_dist = None
        self.a_theta_dist = None
        self.sensitivity_dist = None
        self.capital_dist = None
        self.alpha_dist = None
        self.lambda_dist = None
        self.initial_graph_type = None
        self.model_graph = None
        self.step_target = None
        self.steering_parameters = None
        self.model_data  = None
        self.step_count = None

    def set_model_parameters(self,*,parameterFilePath=None, default=True, **kwargs):
        """
        Load or set model parameters

        :param parameterFlePath: optional, path to parameter file
        :param default: Specify whether default values should be used (True;default)
        :param **kwargs: flexible passing of mode parameters. Only those supported by the model are accepted.
                         If parameters are passed, non-specifed parameters will be set with defaults.

        """
        modelpars = self.__dict__.keys()
        # remove savestate and restart from modelpars
        modelpars = [par for par in modelpars if par not in ['savestate','restart', 'inputs']]

        if parameterFilePath != None:
            with open(parameterFilePath, 'r') as readfile:
                try:
                    self.__dict__ = yaml.safe_load(readfile)
                except yaml.YAMLError as exc:
                    raise SyntaxError(exc)

            for modelpar in modelpars:
                if modelpar not in ['_model_identifier','model_graph','steering_parameters']:
                    if type(self.__dict__[modelpar]) is list:
                        self.__dict__[modelpar] = torch.tensor(self.__dict__[modelpar])
                elif modelpar in ['steering_parameters']:
                    for params in self.steering_parameters.keys():
                        if type(self.steering_parameters[params]) is list:
                            if type(self.steering_parameters[params][0]) is not str:
                                self.steering_parameters[params] = torch.tensor(self.steering_parameters[params])

        else:
            if default:
                for modelpar in modelpars:
                    if modelpar not in ['_model_identifier','model_graph']:
                        self.__dict__[modelpar] = self.default_model_parameters[modelpar]
                self.steering_parameters['npath'] = './'+self._model_identifier+'/agent_data.zarr'
                self.steering_parameters['epath'] = './'+self._model_identifier+'/edge_data'
            else:
                if kwargs:
                    kwpars = kwargs.keys()
                    for kwpar in kwpars:
                        if kwpar in modelpars:
                            self.__dict__[kwpar] = kwargs[kwpar]
                        else:
                            raise ValueError(f'Specified parameter {kwpar} is not supported')
                    for modelpar in modelpars:
                        if (modelpar not in kwpars) and (modelpar not in ['_model_identifier','model_graph']):
                            self.__dict__[modelpar] = self.default_model_parameters[modelpar]
                else:
                    raise ValueError('default model has not been selected, but no model parameters have been supplied')

        if self.restart:
            self.step_count = self.restart - 1

    def initialize_model(self):
        """
        convenience fucntion to create network and initiliize agent properties in correct order, thereby initializing a model
        """
        self.create_network()
        self.initialize_agent_properties()
        self.initialize_model_properties()
        weight_update(self.model_graph, self.steering_parameters['homophily_parameter'], self.steering_parameters['characteristic_distance'], self.steering_parameters['truncation_weight'])
        data_collection(self.model_graph, timestep = 0, npath = self.steering_parameters['npath'], epath = self.steering_parameters['epath'], ndata = self.steering_parameters['ndata'],
                    edata = self.steering_parameters['edata'], format = self.steering_parameters['format'], mode = self.steering_parameters['mode'])

        self.inputs = {
            self.step_count: {
                'model_graph': copy.deepcopy(self.model_graph),  # it gets updated in the step function
                'model_data': self.model_data,  # it does not get updated, but it is needed for the step function
                'generator_state': generator.get_state(),
            }
        }


    def create_network(self):
        """
        Create intial network connecting agents. Makes use of intial graph type specified as model parameter
        """

        agent_graph = network_creation(self.number_agents, self.initial_graph_type, seed=1)
        self.model_graph = agent_graph

    def initialize_model_properties(self):
        """
        Initialize model properties.
        Values are initialized as tensors of length corresponding to number of time steps.
        """
        modelTheta = self._initialize_model_theta()
        self.model_data['modelTheta'] = modelTheta

    def _initialize_model_theta(self):
        modelTheta = sample_distribution_tensor(self.steering_parameters['m_theta_dist']['type'],self.steering_parameters['m_theta_dist']['parameters'],self.step_target,round=self.steering_parameters['m_theta_dist']['round'],decimals=self.steering_parameters['m_theta_dist']['decimals'])
        return modelTheta

    def initialize_agent_properties(self):
        """
        initialize and assign agent properties. Note: agents are represented as nodes of the model graph.
        Values are initialized as tensors of length corresponding to number of agents, with vaues subsequently
        being assigned to the nodes.
        """
        agentsCapital = self._initialize_agents_capital()
        agentsAlpha = self._initialize_agents_alpha()
        agentsLam =  self._initialize_agents_lam()
        agentsSigma = self._initialize_agents_sigma()
        agentsTheta = self._initialize_agents_theta()
        agentsSensitivity = self._initialize_agents_sensitivity()
        agentsAdaptTable = self._initialize_agents_adapttable()
        agentsTecLevel, agentsGamma, agentsCost = self._initialize_agents_tec()

        # TODO: add comment explaining what each variable is (here? where?).
        if isinstance(self.model_graph,dgl.DGLGraph):
            self.model_graph.ndata['wealth'] = agentsCapital
            self.model_graph.ndata['alpha'] = agentsAlpha
            self.model_graph.ndata['theta'] = agentsTheta
            self.model_graph.ndata['sensitivity'] = agentsSensitivity
            self.model_graph.ndata['lambda'] = agentsLam
            self.model_graph.ndata['sigma'] = agentsSigma
            self.model_graph.ndata['technology_level'] = agentsTecLevel
            self.model_graph.ndata['gamma'] = agentsGamma
            self.model_graph.ndata['cost'] = agentsCost
            self.model_graph.ndata['a_table'] = agentsAdaptTable
            self.model_graph.ndata['wealth_consumption'] = torch.zeros(self.model_graph.num_nodes())
            self.model_graph.ndata['i_a'] = torch.zeros(self.model_graph.num_nodes())
            self.model_graph.ndata['m'] = torch.zeros(self.model_graph.num_nodes())
            self.model_graph.ndata['zeros'] = torch.zeros(self.model_graph.num_nodes())
            self.model_graph.ndata['ones'] = torch.ones(self.model_graph.num_nodes())
        else:
            raise RuntimeError('model graph must be a defined DGLgraph object. Consder running `create_network` before initializing agent properties')


    def _initialize_agents_adapttable(self):
        """
        Initialize agents adaptation measure knowledge, currently uniform.
        """
        agentsAdaptTable =torch.stack([self.steering_parameters['adapt_m'],self.steering_parameters['adapt_cost']]).repeat(self.number_agents,1,1)
        return agentsAdaptTable

    def _initialize_agents_theta(self):
        """
        Initialize agents theta as a 1d tensor sampled from the specified initial theta distribution
        """
        agentsTheta = sample_distribution_tensor(self.a_theta_dist['type'],self.a_theta_dist['parameters'],self.number_agents,round=self.a_theta_dist['round'],decimals=self.a_theta_dist['decimals'])
        return agentsTheta

    def _initialize_agents_sensitivity(self):
        """
        Initialize agents sensitivity as a 1d tensor sampled from the specified initial sensitivity distribution
        """
        agentsSensitivity = sample_distribution_tensor(self.sensitivity_dist['type'],self.sensitivity_dist['parameters'],self.number_agents,round=self.sensitivity_dist['round'],decimals=self.sensitivity_dist['decimals'])
        return agentsSensitivity

    def _initialize_agents_capital(self):
        """
        Initialize agents captial as a 1d tensor sampled from the specified intial capital distribution
        """
        agentsCapital = sample_distribution_tensor(self.capital_dist['type'],self.capital_dist['parameters'],self.number_agents,round=self.capital_dist['round'],decimals=self.capital_dist['decimals'])
        return agentsCapital

    def _initialize_agents_alpha(self):
        """
        Initialize agents alpha as a 1d tensor sampled from the specified intial alpha distribution
        """
        agentsAlpha = sample_distribution_tensor(self.alpha_dist['type'],self.alpha_dist['parameters'],self.number_agents,round=self.alpha_dist['round'],decimals=self.alpha_dist['decimals'])
        return agentsAlpha

    def _initialize_agents_lam(self):
        """
        Initialize agents lambda as a 1d tensor sampled from the specified intial lambda distribution
        """
        agentsLam = sample_distribution_tensor(self.lambda_dist['type'],self.lambda_dist['parameters'],self.number_agents,round=self.lambda_dist['round'],decimals=self.lambda_dist['decimals'])
        return agentsLam

    def _initialize_agents_sigma(self):
        """
        Initialize agents sigma as a 1d tensor
        """
        agentsSigma = sample_distribution_tensor(self.sigma_dist['type'],self.sigma_dist['parameters'],self.number_agents,round=self.sigma_dist['round'],decimals=self.sigma_dist['decimals'])
        return agentsSigma

    def _initialize_agents_tec(self):
        """
        Initialize the agents technology level distribution as 1d tensor sampled from the specified intial technology level distribution.
        Initialize agents gamma and cost distributions according to their technology level and the spefied initial gamma and cost
        values associated with that tech level
        """
        agentsTecLevel = sample_distribution_tensor(self.technology_dist['type'],self.technology_dist['parameters'],self.number_agents,round=self.technology_dist['round'],decimals=self.technology_dist['decimals'])
        agentsGamma = torch.zeros(self.number_agents)
        agentsCost = torch.zeros(self.number_agents)
        for i in range(len(self.technology_levels)):
            technology_mask = agentsTecLevel == i
            agentsGamma[technology_mask] = self.gamma_vals[i]
            agentsCost[technology_mask] = self.cost_vals[i]
        return agentsTecLevel, agentsGamma, agentsCost

    def step(self):
        try:
            self.step_count +=1
            print(f'performing step {self.step_count} of {self.step_target}')
            ptm_step(self.model_graph, self.model_data, self.step_count, self.steering_parameters)

            # store each step
            self.inputs[self.step_count] = {
                'model_graph': copy.deepcopy(self.model_graph),
                'model_data': self.model_data,
                'generator_state': generator.get_state(),
            }

        except:
            #TODO add model dump here. Also check against previous save to avoid overwriting
            raise RuntimeError(f'execution of step failed for step {self.step_count}')

    def run(self):
        """ run the model for each step until the step_target is reached."""

        self.model_graph = copy.deepcopy(self.inputs[self.step_count]['model_graph'])
        self.model_data = self.inputs[self.step_count]['model_data']
        generator.set_state(self.inputs[self.step_count]['generator_state'])

        while self.step_count < self.step_target:
            self.step()

        if self.savestate:
            path = f'./{self._model_identifier}'
            _save_model(path, self.inputs)


def _save_model(path, dictionary):
    """ save the model_graph and generator_state in files."""

    # save all the model_graphs
    graph_list = [val.get('model_graph') for val in dictionary.values()]
    save_graphs(str(Path(path) / "model_graphs.bin"), graph_list)

    # save the generator_state
    generator_list = [val.get('generator_state') for val in dictionary.values()]
    with open(Path(path) / "generator_state.bin", 'wb') as file:
        pickle.dump(generator_list, file)

    # save model_data
    model_data = dictionary.get(0).get('model_data')
    with open(Path(path) / "model_data.bin", 'wb') as file:
        pickle.dump(model_data, file)


def _load_model(path):
    graph_list = load_graphs(str(Path(path) / "model_graphs.bin"))[0]

    with open(Path(path) / "generator_state.bin", 'rb') as file:
        generator_list = pickle.load(file)

    with open(Path(path) / "model_data.bin", 'rb') as file:
        model_data = pickle.load(file)

    # create a dictionary with the model_graphs and generator_state
    inputs = {}
    for i, graph in enumerate(graph_list):
        inputs[i] = {
            'model_graph': graph,
            'model_data': model_data,
            'generator_state': generator_list[i],
        }

    return inputs
