import copy
import dgl
import torch
import pickle
import logging

from pathlib import Path
from dgl.data.utils import save_graphs, load_graphs

from dgl_ptm.network.network_creation import network_creation
from dgl_ptm.model.step import ptm_step
from dgl_ptm.agentInteraction.weight_update import weight_update
from dgl_ptm.config import Config, CONFIG
from dgl_ptm.util.network_metrics import average_degree

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
        dist = torch.gather(torch.Tensor(distParameters[1]), 0, torch.multinomial(torch.tensor(distParameters[0]), nSamples, replacement=True))
    elif type == 'truncnorm':
        # distParameters are mean, standard deviation, min, and max. cdf(x)=(1+erf(x/2^0.5))/2. cdf^-1(x)=2^0.5*erfinv(2*x-1).
        trunc_val_min = (distParameters[2]-distParameters[0])/distParameters[1]
        trunc_val_max = (distParameters-distParameters[0])/distParameters[1]
        cdf_min = (1 + torch.erf(trunc_val_min / torch.sqrt(torch.tensor(2.0))))/2
        cdf_max = (1 + torch.erf(trunc_val_max / torch.sqrt(torch.tensor(2.0))))/2

        uniform_samples = torch.rand(size)
        sample_ppf = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 *(cdf_min + (cdf_max - cdf_min) * uniform_samples) - 1)

        dist = distParameters[0] + distParameters[1] * sample_ppf

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

    def __init__(self,*, model_identifier):
        """
        restore from a checkpoint or create a PVT model instance.
        Checks whether a model indentifier has been specified.

        param: model_identifier: str, required. Identifier for the model. Used to save and load model states.

        """

        super().__init__(model_identifier = model_identifier)

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
        self.initial_graph_args = CONFIG.initial_graph_args
        self.model_graph = CONFIG.model_graph
        self.step_count = CONFIG.step_count
        self.step_target = CONFIG.step_target
        self.checkpoint_period = CONFIG.checkpoint_period
        self.milestones = CONFIG.milestones
        self.steering_parameters = CONFIG.steering_parameters

        # Code version.
        self.version = Path('version.md').read_text().splitlines()[0]


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

        # save updated config to yaml files
        cfg_filename = parent_dir / f'{self._model_identifier}.yaml'
        cfg.to_yaml(cfg_filename)
        cfg_filename_step = parent_dir / f'{self._model_identifier}_{self.step_count}.yaml'
        cfg.to_yaml(cfg_filename_step)
        logger.warning(f'The model parameters are saved to {cfg_filename} and {cfg_filename_step}.')

    def initialize_model(self):
        """
        convenience fucntion to create network and initiliize agent properties in correct order, thereby initializing a model
        """
        torch.manual_seed(self.seed)
        self.create_network()
        self.initialize_agent_properties()
        self.model_graph = self.model_graph.to(self.device)
        self.initialize_model_properties()
        self.steering_parameters['modelTheta'] = self.steering_parameters['modelTheta'].to(self.device)

        weight_update(self.model_graph, self.device, self.steering_parameters['homophily_parameter'], self.steering_parameters['characteristic_distance'], self.steering_parameters['truncation_weight'])
        #data_collection(self.model_graph, timestep = 0, npath = self.steering_parameters['npath'], epath = self.steering_parameters['epath'], ndata = self.steering_parameters['ndata'],
                    #edata = self.steering_parameters['edata'], format = self.steering_parameters['format'], mode = self.steering_parameters['mode'])

        # store random generator state
        self.generator_state = generator.get_state()

        # number of edges(links) in the network
        self.number_of_edges = self.model_graph.number_of_edges()
        self.average_degree = average_degree(self.model_graph)

    def create_network(self):
        """
        Create intial network connecting agents. Makes use of intial graph type specified as model parameter
        """

        agent_graph = network_creation(self.number_agents, self.initial_graph_type, **self.initial_graph_args)
        #this should fix issues with execution oon GPU (fix by VG)
        self.model_graph = agent_graph.to(self.device)
        #self.model_graph = agent_graph


    def initialize_model_properties(self):
        """
        Initialize model properties.
        Values are initialized as tensors of length corresponding to number of time steps.
        """
        modelTheta = self._initialize_model_theta()
        self.steering_parameters['modelTheta'] = modelTheta
        #attachProb = self._initialize_attach_prob()
        #self.steering_parameters['attachProb'] = attachProb

    def _initialize_model_theta(self):
        modelTheta = sample_distribution_tensor(self.steering_parameters['m_theta_dist']['type'],self.steering_parameters['m_theta_dist']['parameters'],self.step_target,round=self.steering_parameters['m_theta_dist']['round'],decimals=self.steering_parameters['m_theta_dist']['decimals'])
        return modelTheta

    def _initialize_attach_prob(self):
        attachProb = sample_distribution_tensor(self.steering_parameters['m_attach_dist']['type'],self.steering_parameters['m_attach_dist']['parameters'],self.step_target,round=self.steering_parameters['m_attach_dist']['round'],decimals=self.steering_parameters['m_attach_dist']['decimals'])
        return attachProb

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
          #send to device!!
            self.model_graph.ndata['wealth'] = agentsCapital.to(self.device)
            self.model_graph.ndata['alpha'] = agentsAlpha.to(self.device)
            self.model_graph.ndata['theta'] = agentsTheta.to(self.device)
            self.model_graph.ndata['sensitivity'] = agentsSensitivity.to(self.device)
            self.model_graph.ndata['lambda'] = agentsLam.to(self.device)
            self.model_graph.ndata['sigma'] = agentsSigma.to(self.device)
            self.model_graph.ndata['tec'] = agentsTecLevel.to(self.device)
            self.model_graph.ndata['gamma'] = agentsGamma.to(self.device)
            self.model_graph.ndata['cost'] = agentsCost.to(self.device)
            self.model_graph.ndata['a_table'] = agentsAdaptTable.to(self.device)
            self.model_graph.ndata['wealth_consumption'] = torch.zeros(self.model_graph.num_nodes()).to(self.device)
            self.model_graph.ndata['i_a'] = torch.zeros(self.model_graph.num_nodes()).to(self.device)
            self.model_graph.ndata['m'] = torch.zeros(self.model_graph.num_nodes()).to(self.device)
            self.model_graph.ndata['zeros'] = torch.zeros(self.model_graph.num_nodes()).to(self.device)
            self.model_graph.ndata['ones'] = torch.ones(self.model_graph.num_nodes()).to(self.device)
        else:
            raise RuntimeError('model graph must be a defined as DGLgraph object. Consider running `create_network` before initializing agent properties')


    def _initialize_agents_adapttable(self):
        """
        Initialize agent adaptation measure knowledge, currently uniform.
        """
        agentsAdaptTable =torch.stack([self.steering_parameters['adapt_m'],self.steering_parameters['adapt_cost']]).repeat(self.number_agents,1,1)
        return agentsAdaptTable

    def _initialize_agents_theta(self):
        """
        Initialize agent theta as a 1d tensor sampled from the specified initial theta distribution
        """
        agentsTheta = sample_distribution_tensor(self.a_theta_dist['type'],self.a_theta_dist['parameters'],self.number_agents,round=self.a_theta_dist['round'],decimals=self.a_theta_dist['decimals'])
        return agentsTheta

    def _initialize_agents_sensitivity(self):
        """
        Initialize agent sensitivity as a 1d tensor sampled from the specified initial sensitivity distribution
        """
        agentsSensitivity = sample_distribution_tensor(self.sensitivity_dist['type'],self.sensitivity_dist['parameters'],self.number_agents,round=self.sensitivity_dist['round'],decimals=self.sensitivity_dist['decimals'])
        return agentsSensitivity

    def _initialize_agents_capital(self):
        """
        Initialize agent captial as a 1d tensor sampled from the specified initial capital distribution
        """
        agentsCapital = sample_distribution_tensor(self.capital_dist['type'],self.capital_dist['parameters'],self.number_agents,round=self.capital_dist['round'],decimals=self.capital_dist['decimals'])
        return agentsCapital

    def _initialize_agents_alpha(self):
        """
        Initialize agent alpha as a 1d tensor sampled from the specified initial alpha distribution
        """
        agentsAlpha = sample_distribution_tensor(self.alpha_dist['type'],self.alpha_dist['parameters'],self.number_agents,round=self.alpha_dist['round'],decimals=self.alpha_dist['decimals'])
        return agentsAlpha

    def _initialize_agents_lam(self):
        """
        Initialize agent lambda as a 1d tensor sampled from the specified initial lambda distribution
        """
        agentsLam = sample_distribution_tensor(self.lambda_dist['type'],self.lambda_dist['parameters'],self.number_agents,round=self.lambda_dist['round'],decimals=self.lambda_dist['decimals'])
        return agentsLam

    def _initialize_agents_sigma(self):
        """
        Initialize agent sigma as a 1d tensor

        """
        agentsSigma = sample_distribution_tensor(self.sigma_dist['type'],self.sigma_dist['parameters'],self.number_agents,round=self.sigma_dist['round'],decimals=self.sigma_dist['decimals'])
        return agentsSigma

    def _initialize_agents_tec(self):
        """
        Initialize the agents technology level distribution as 1d tensor sampled from the specified intial technology level distribution.
        Initialize agents gamma and cost distributions according to their technology level and the speci fied initial gamma and cost
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
            print(f'performing step {self.step_count} of {self.step_target}')
            ptm_step(self.model_graph, self.device, self.step_count, self.steering_parameters)
            self.step_count +=1

            # number of edges(links) in the network
            self.number_of_edges = self.model_graph.number_of_edges()
            self.average_degree = average_degree(self.model_graph)

        except:
            #TODO add model dump here. Also check against previous save to avoid overwriting
            raise RuntimeError(f'execution of step failed for step {self.step_count}')

    def run(self, restart=False):
        """
        run the model for each step until the step_target is reached.

        param: restart: boolean or int or a pair of ints, optional.
        If True, the model is run from last checkpoint,
        if an int, the model is run from the first milestone at that step,
        if a pair of ints, the model is run from that milestone at that step.
        Default False.
        """

        self.inputs = None
        if isinstance(restart, bool):
            if restart:
                self.inputs = _load_model(f'./{self._model_identifier}')
        elif isinstance(restart, int):
            self.inputs = _load_model(f'./{self._model_identifier}/milestone_{restart}')
        elif isinstance(restart, tuple):
            self.inputs = _load_model(f'./{self._model_identifier}/milestone_{restart[0]}_{restart[1]}')

        if self.inputs:
            self.model_graph = copy.deepcopy(self.inputs["model_graph"])
            #self.model_data = self.inputs["model_data"]
            self.generator_state = self.inputs["generator_state"]
            self.step_count = self.inputs["step_count"]

        generator.set_state(self.generator_state)

        while self.step_count < self.step_target:
            self.step()

            # save the model state every step reported by checkpoint_period and at specific milestones.
            # checkpoint saves overwrite the previous checkpoint; milestone get unique folders.
            save_checkpoint = 0 < self.checkpoint_period and self.step_count % self.checkpoint_period == 0
            save_milestone = self.milestones and self.step_count in self.milestones
            if save_checkpoint or save_milestone:
                self.inputs = {
                    'model_graph': copy.deepcopy(self.model_graph),
                    #'model_data': copy.deepcopy(self.model_data),
                    'generator_state': generator.get_state(),
                    'step_count': self.step_count,
                    'code_version': self.version
                }

                # Note that a sinlge step could be both a checkpoint and a milestone.
                # The checkpoint could be necessary to restore a crashed process while
                # the milestone is required output.
                if save_checkpoint:
                    _save_model(f'./{self._model_identifier}', self.inputs)
                if save_milestone:
                    milestone_path = _make_path_unique(f'./{self._model_identifier}/milestone_{self.step_count}')
                    _save_model(milestone_path, self.inputs)

def _make_path_unique(path):
    if Path(path).exists():
        incr = 1
        def add_incr(path, incr): return f'{path}_{incr}'
        while Path(add_incr(path, incr)).exists(): incr += 1
        path = add_incr(path, incr)
    return path

def _save_model(path, inputs):
    """ save the model_graph, generator_state and code_version in files."""

    # save the model_graph with a label
    graph_labels = {'step_count': torch.tensor([inputs["step_count"]])}
    save_graphs(str(Path(path) / "model_graph.bin"), inputs["model_graph"], graph_labels)

    # save the generator_state
    with open(Path(path) / "generator_state.bin", 'wb') as file:
        pickle.dump([inputs["generator_state"], inputs["step_count"]], file)

    # save model_data
    #with open(Path(path) / "model_data.bin", 'wb') as file:
    #    pickle.dump([inputs["model_data"], inputs["step_count"]], file)

    # save the code version
    with open(Path(path) / "version.md", 'w') as file:
        file.writelines([inputs["code_version"] + '\n', f'step={inputs["step_count"]}\n'])


def _load_model(path):
    # Load model graph
    path_model_graph = Path(path) / "model_graph.bin"
    if not path_model_graph.is_file():
        raise ValueError(f'The path {path_model_graph} is not a file.')

    graph, graph_labels = load_graphs(str(path_model_graph))
    graph = graph[0]
    graph_step = graph_labels['step_count'].tolist()[0]

    # Load generator_state
    path_generator_state = Path(path) / "generator_state.bin"
    if not path_generator_state.is_file():
        raise ValueError(f'The path {path_generator_state} is not a file.')

    with open(path_generator_state, 'rb') as file:
        generator, generator_step = pickle.load(file)

    # Load model_data
    #path_model_data = Path(path) / "model_data.bin"
    #if not path_model_data.is_file():
    #    raise ValueError(f'The path {path_model_data} is not a file.')

    #with open(path_model_data, 'rb') as file:
    #    data, data_step = pickle.load(file)

    # Load code version
    path_code_version = Path(path) / "version.md"
    if not path_code_version.is_file():
        raise ValueError(f'The path {path_code_version} is not a file.')

    with open(path_code_version, 'r') as file:
        code_version = file.readlines()[0]

    # Check if graph_step, generator_step and data_step are the same
    if graph_step != generator_step: #or graph_step != data_step:
        msg = 'The step count in the model_graph and generator_state are not the same.'# and model_data are not the same.'
        raise ValueError(msg)
    
    # Check if the saved version and current code version are the same
    version = Path('version.md').read_text().splitlines()[0]
    if code_version != version:
        logger.warning(f'Warning: loading model generated using earlier code version: {code_version}.')

    # Show which step is loaded
    logger.warning(f'Loading model state from step {generator_step}.')

    inputs = {
        'model_graph': graph,
        #'model_data': data,
        'generator_state': generator,
        'step_count': generator_step,
        'code_version': code_version
    }
    return inputs
