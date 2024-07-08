import copy
from pathlib import Path
import dgl
import torch
import pickle
import logging
from pathlib import Path

from dgl_ptm.network.network_creation import network_creation
from dgl_ptm.model.step import ptm_step
from dgl_ptm.agentInteraction.weight_update import weight_update
from dgl_ptm.model.data_collection import data_collection
from dgl.data.utils import save_graphs, load_graphs
from dgl_ptm.config import Config, CONFIG

# Set the seed of the random number generator
# this is global and will affect all random number generators
generator = torch.manual_seed(0)

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
    def __init__(self,*, model_identifier, restart=False, savestate=1):
        """
        restore from a savestate or create a PVT model instance.
        Checks whether a model indentifier has been specified.

        param: model_identifier: str, required. Identifier for the model. Used to save and load model states.
        param: restart: boolean, optional. If True, the model is run from last
        saved step. Default False.
        param: savestate: int, optional. If provided, the model state is saved
        on this frequency. Default is 1 i.e. every time step.
        """
        super().__init__(model_identifier = model_identifier)
        self.restart = restart
        self.savestate = savestate

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

        if cfg.model_identifier != self._model_identifier:
            logger.warning(f'A model identifier has been set as "{self._model_identifier}". '
                           f'But another identifier "{cfg.model_identifier}" is provided (perhaps by default). '
                           f'The identifier "{self._model_identifier}" will be used.')

        cfg.model_identifier = self._model_identifier # see config.py for why cfg.model_identifier

        # update model parameters/ attributes
        cfg_dict = cfg.model_dump(by_alias=True, warnings=False)
        for key, value in cfg_dict.items():
            setattr(self, key, value)

        # Correct the paths
        parent_dir = "." / Path(self._model_identifier)
        parent_dir.mkdir(parents=True, exist_ok=True)
        self.steering_parameters['npath'] = str(parent_dir / Path(cfg.steering_parameters.npath))
        self.steering_parameters['epath'] = str(parent_dir / Path(cfg.steering_parameters.epath))

        # save updated config to yaml file
        cfg_filename = parent_dir / f'{self._model_identifier}.yaml'
        cfg.to_yaml(cfg_filename)
        logger.warning(f'The model parameters are saved to {cfg_filename}.')

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

        # store random generator state
        self.generator_state = generator.get_state()

        # number of edges(links) in the network
        self.number_of_edges = self.model_graph.number_of_edges()

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
            ptm_step(self.model_graph, self.device, self.model_data, self.step_count, self.steering_parameters)

            # number of edges(links) in the network
            self.number_of_edges = self.model_graph.number_of_edges()

        except:
            #TODO add model dump here. Also check against previous save to avoid overwriting
            raise RuntimeError(f'execution of step failed for step {self.step_count}')

    def run(self):
        """ run the model for each step until the step_target is reached."""

        if self.restart:
            self.inputs = _load_model(f'./{self._model_identifier}')
            self.model_graph = copy.deepcopy(self.inputs["model_graph"])
            self.model_data = self.inputs["model_data"]
            self.generator_state = self.inputs["generator_state"]
            self.step_count = self.inputs["step_count"]

        generator.set_state(self.generator_state)

        while self.step_count < self.step_target:
            self.step()

            # save the model state every step reported by savestate
            if self.savestate and self.step_count % self.savestate == 0:
                self.inputs = {
                    'model_graph': copy.deepcopy(self.model_graph),
                    'model_data': copy.deepcopy(self.model_data),
                    'generator_state': generator.get_state(),
                    'step_count': self.step_count
                }
                _save_model(f'./{self._model_identifier}', self.inputs)


def _save_model(path, inputs):
    """ save the model_graph, generator_state and model_data in files."""

    # save the model_graph with a label
    graph_label = {'step_count': torch.tensor([inputs["step_count"]])}
    save_graphs(str(Path(path) / "model_graphs.bin"), inputs["model_graph"], graph_label)

    # save the generator_state
    with open(Path(path) / "generator_state.bin", 'wb') as file:
        pickle.dump([inputs["generator_state"], inputs["step_count"]], file)

    # save model_data
    with open(Path(path) / "model_data.bin", 'wb') as file:
        pickle.dump([inputs["model_data"], inputs["step_count"]], file)


def _load_model(path):
    # Load model graphs
    path_model_graph = Path(path) / "model_graphs.bin"
    if not path_model_graph.is_file():
        raise ValueError(f'The path {path_model_graph} is not a file.')

    graph, graph_lebel = load_graphs(str(path_model_graph))
    graph = graph[0]
    graph_step = graph_lebel['step_count'].tolist()[0]

    # Load generator_state
    path_generator_state = Path(path) / "generator_state.bin"
    if not path_generator_state.is_file():
        raise ValueError(f'The path {path_generator_state} is not a file.')

    with open(path_generator_state, 'rb') as file:
        generator, generator_step = pickle.load(file)

    # Load model_data
    path_model_data = Path(path) / "model_data.bin"
    if not path_model_data.is_file():
        raise ValueError(f'The path {path_model_data} is not a file.')

    with open(path_model_data, 'rb') as file:
        data, data_step = pickle.load(file)

    # Check if graph_step, generator_step and data_step are the same
    if graph_step != generator_step or graph_step != data_step:
        msg = 'The step count in the model_graph, generator_state and model_data are not the same.'
        raise ValueError(msg)

    # Show which step is loaded
    logger.warning(f'Loading model state from step {data_step}.')

    inputs = {
        'model_graph': graph,
        'model_data': data,
        'generator_state': generator,
        'step_count': data_step
    }
    return inputs
