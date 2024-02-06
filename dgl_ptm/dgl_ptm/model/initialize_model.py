import dgl
import networkx as nx
import torch
import yaml
import logging
from pathlib import Path

from dgl_ptm.network.network_creation import network_creation
from dgl_ptm.model.step import ptm_step
from dgl_ptm.agentInteraction.weight_update import weight_update
from dgl_ptm.model.data_collection import data_collection
from dgl_ptm.config import Config, CONFIG

logger = logging.getLogger(__name__)

def sample_distribution_tensor(type, distParameters, nSamples, round=False, decimals=None):
    """
    create and return samples from different distributions

    :param type: Type of distribution to sample
    :param distParameters: array of parameters as required/supported by requested distribution type
    :param nSamples: number of samples to return (as 1d tensor)
    :param round: optional, whether the samples are to be rounded
    :param decimals: optional, required if round is specified. decimal places to round to
    """
    # check if each item in distParameters are torch tensors, if not convert them
    for i, item in enumerate(distParameters):
        # if item has dtype NoneType, raise error
        if item != None and not isinstance(item, torch.Tensor):
                distParameters[i] = torch.tensor(item)

    if not isinstance(nSamples, torch.Tensor):
        nSamples = torch.tensor(nSamples)

    if type == 'uniform':
        dist = torch.distributions.uniform.Uniform(distParameters[0], distParameters[1]).sample([nSamples])
    elif type == 'normal':
        dist = torch.distributions.normal.Normal(distParameters[0], distParameters[1]).sample([nSamples])
    elif type == 'bernoulli':
        dist = torch.distributions.bernoulli.Bernoulli(probs=distParameters[0],logits=distParameters[1],validate_args=None).sample([nSamples])
    elif type == 'multinomial':
        dist = torch.gather(distParameters[1], 0, torch.multinomial(distParameters[0], nSamples, replacement=True))

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

    def __init__(self,*, model_identifier=None, restart=False, savestate=None):
        """
        restore from a savestate (TODO) or create a PVT model instance.
        Checks whether a model indentifier has been specified.
        """

        if restart:
            if savestate==None:
                raise ValueError('When restarting a simulation an intial savestate must be supplied')
            else:
                #TODO implement restart
                pass
        else:
            super().__init__(model_identifier = model_identifier)
            if self._model_identifier == None:
                raise ValueError('A model identifier must be specified')

            # default values
            self.device = CONFIG.device
            self.number_agents = CONFIG.number_agents
            self.gamma_vals = CONFIG.gamma_vals
            self.sigma_dist = CONFIG.sigma_dist
            self.cost_vals = CONFIG.cost_vals
            self.technology_levels = CONFIG.technology_levels
            self.technology_dist = CONFIG.technology_dist
            self.a_theta_dist = CONFIG.a_theta_dist
            self.sensitivity_dist = CONFIG.sensitivity_dist
            self.capital_dist = CONFIG.capital_dist
            self.alpha_dist = CONFIG.alpha_dist
            self.lambda_dist = CONFIG.lambda_dist
            self.initial_graph_type = CONFIG.initial_graph_type
            self.model_graph = CONFIG.model_graph
            self.step_count = CONFIG.step_count
            self.step_target = CONFIG.step_target
            self.steering_parameters = CONFIG.steering_parameters
            self.model_data = CONFIG.model_data

    def set_model_parameters(self, *, parameterFilePath=None, **kwargs):
        """
        Load or set model parameters

        :param parameterFlePath: optional, path to parameter file. If not,  default values are used.
        :param **kwargs: flexible passing of mode parameters. Only those supported by the model are accepted.
                         If parameters are passed, non-specifed parameters will be set with defaults.

        """

        cfg = CONFIG # default values

        if parameterFilePath:
            cfg = Config.from_yaml(parameterFilePath)

        if kwargs:
            cfg = Config.from_dict(kwargs)

        # if both parameterFilePath and kwargs are set, combine them into one.
        # if fields are duplicated, kwargs will overwrite parameterFilePath
        if parameterFilePath and kwargs:
            cfg = Config.from_yaml(parameterFilePath)
            for key, value in kwargs.items():
                setattr(cfg, key, value)
            logger.warning(
                'model parameters have been provided via parameterFilePath and **kwargs. '
                '**kwargs will overwrite parameterFilePath'
                )

        if parameterFilePath is None and not kwargs:
            logger.warning('no model parameters have been provided, Default values are used')

        cfg.model_identifier = self._model_identifier # see config.py for why cfg.model_identifier

        # save updated config to yaml file
        cfg_filename = f'./{self._model_identifier}.yaml'
        cfg.to_yaml(cfg_filename)
        logger.warning(f'We have saved the model parameters to {cfg_filename}.')

        # update model parameters
        self.__dict__ = cfg.model_dump(by_alias=True, warnings=False)

        parent_dir = "." / Path(self._model_identifier)
        self.steering_parameters['npath'] = str(parent_dir / Path(cfg.steering_parameters.npath))
        self.steering_parameters['epath'] = str(parent_dir / Path(cfg.steering_parameters.epath))


    def initialize_model(self):
        """
        convenience fucntion to create network and initiliize agent properties in correct order, thereby initializing a model
        """
        self.create_network()
        self.initialize_agent_properties()
        self.model_graph = self.model_graph.to(self.device)
        self.initialize_model_properties()
        self.model_data['modelTheta'] = self.model_data['modelTheta'].to(self.device)

        weight_update(self.model_graph, self.device, self.steering_parameters['homophily_parameter'], self.steering_parameters['characteristic_distance'], self.steering_parameters['truncation_weight'])
        data_collection(self.model_graph, timestep = 0, npath = self.steering_parameters['npath'], epath = self.steering_parameters['epath'], ndata = self.steering_parameters['ndata'],
                    edata = self.steering_parameters['edata'], format = self.steering_parameters['format'], mode = self.steering_parameters['mode'])

    def create_network(self):
        """
        Create intial network connecting agents. Makes use of intial graph type specified as model parameter
        """

        agent_graph = network_creation(self.number_agents, self.initial_graph_type)
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
            ptm_step(self.model_graph,self.device,self.model_data,self.step_count,self.steering_parameters)
        except:
            #TODO add model dump here. Also check against previous save to avoid overwriting


            raise RuntimeError(f'execution of step failed for step {self.step_count}')


    def run(self):
        while self.step_count < self.step_target:
            print(f'performing step {self.step_count+1} of {self.step_target}')
            self.step()
