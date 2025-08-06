"""This module contains the model class and functions to initialize the model."""

import copy
import logging
import pickle
from pathlib import Path

import dgl
import torch
from dgl.data.utils import load_graphs, save_graphs

from dgl_ptm.agentInteraction.weight_update import weight_update_sveir
from dgl_ptm.config import Config, SVEIRCONFIG, SVEIRConfig
from dgl_ptm.model.step import sveir_step
from dgl_ptm.network.network_creation import network_creation
from dgl_ptm.util.network_metrics import average_degree, node_degree
from dgl_ptm.environment.grid_creation import grid_creation
from dgl_ptm.environment.grid_assignment import grid_assignment
from dgl_ptm.util.utils import sample_distribution_tensor
from dgl_ptm.model.policy_engine import *

# Set the seed of the random number generator
# this is global and will affect all random number generators
generator = torch.manual_seed(0)

logger = logging.getLogger(__name__)

def sample_distribution(distribution, n_samples):
    """Sample from a distribution."""
    return sample_distribution_tensor(
        distribution['type'],
        distribution['parameters'],
        n_samples,
        round = distribution['round'],
        decimals = distribution['decimals']
        )


class Model:
    """Abstract model class."""

    def __init__(self, model_identifier = None, root_path = '.'):
        """Initialize the model class.

        Args:
            model_identifier: an identifier used in paths. Defaults to None.
            root_path: path to working directory of the model. Defaults to '.'.
        """
        self._model_identifier = model_identifier
        self.root_path = root_path
        self.model_dir = self.root_path / Path(self._model_identifier)

        # Step count.
        # Note that the config no longer contains the step count:
        # the config is determined before a starting run;
        # the step count may not be correct when loading a config to continue a run
        # (whether restoring a run after a crash or continuing from a milestone).
        self.step_count = 0

    def save_model_parameters(self, overwrite = False):
        """Save model parameters to a yaml file."""
        cfg_filename = f'{self.model_dir}/{self._model_identifier}_{self.step_count}'
        if overwrite:
            cfg_filename = f'{cfg_filename}.yaml'
        else:
            cfg_filename = _make_path_unique(cfg_filename, '.yaml')
        self.config.to_yaml(cfg_filename)
        logger.warning(f'The model parameters are saved to {cfg_filename}.')

    def create_network(self):
        """Create network connecting agents."""
        raise NotImplementedError('network creation is not implemented for this class.')

    def step(self):
        """Perform a single step of the model."""
        raise NotImplementedError('step function is not implemented for this class.')

    def run(self):
        """Run the model for each step until the step_target is reached."""
        raise NotImplementedError('run method is not implemented for this class.')


class SVEIRModel(Model):

    def __init__(self, *, model_identifier, root_path = '.'):
        """Create a new PVT model instance.

        Checks whether a model identifier has been specified.

        param: model_identifier: str, required. Identifier for the model. Used
        to save and load model states.
        param: root_path: str, optional. Root path where to store the model data
        and states.
        """
        super().__init__(model_identifier = model_identifier, root_path = root_path)

        # Attach config.
        self.config = copy.deepcopy(SVEIRCONFIG)
        self.steering_parameters = self.config.steering_parameters.__dict__
        self.graph = None
        self.step_first = -1

        # Process version.
        version_path = Path(__file__).resolve().parents[2] / 'version.md'
        self.version = version_path.read_text().splitlines()[0]

    def set_model_parameters(self, *, parameter_file_path=None, overwrite = False, **kwargs):  # noqa: E501
        """Load or set model parameters.

        :param parameterFlePath: optional, path to parameter file. If not,
            default values are used.
        :param **kwargs: flexible passing of mode parameters. Only those
                         supported by the model are accepted. If parameters are
                         passed, non-specifed parameters will be set with
                         defaults.

        """
        cfg = SVEIRCONFIG # default values

        if parameter_file_path:
            cfg = Config.from_yaml(parameter_file_path)
            if kwargs:
                # if both parameter_file_path and kwargs are set, combine them
                # into one. if fields are duplicated, kwargs will overwrite
                # parameter_file_path
                for key, value in kwargs.items():
                    if isinstance(value, dict):
                        # Special recursive case for steering_parameters: this
                        # makes sure to append to, not overwrite, the steering
                        # parameters.
                        for subkey, subvalue in value.items():
                            setattr(cfg.__dict__[key], subkey, subvalue)
                    else:
                        setattr(cfg, key, value)
                logger.warning(
                    'model parameters have been provided via '
                    'parameter_file_path and **kwargs. '
                    '**kwargs will overwrite parameter_file_path'
                    )
        elif kwargs:
            cfg = SVEIRConfig.from_dict(kwargs)

        if parameter_file_path is None and not kwargs:
            logger.warning(
                'no model parameters have been provided, Default values are used'
                )

        if cfg.model_identifier != self._model_identifier:
            logger.warning(
                f'A model identifier has been set as "{self._model_identifier}". '
                f'But the identifier "{cfg.model_identifier}" is provided by default. '
                f'The identifier "{self._model_identifier}" will be used.'
                )

        # see config.py for why cfg.model_identifier
        cfg.model_identifier = self._model_identifier
        self.config.model_identifier = self._model_identifier

        # update model parameters/ attributes
        cfg_dict = cfg.model_dump(by_alias=True, warnings=False)
        for key, value in cfg_dict.items():
            setattr(self.config, key, value)
        self.steering_parameters = self.config.steering_parameters.__dict__

        # Correct the paths
        self.model_dir = self.root_path / Path(self._model_identifier)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        npath = Path(self.config.steering_parameters.npath)
        self.steering_parameters['npath'] = str(self.model_dir / npath)
        epath = Path(self.config.steering_parameters.epath)
        self.steering_parameters['epath'] = str(self.model_dir / epath)

        # Save updated config to yaml file.
        self.save_model_parameters(overwrite=True)

    def initialize_model(self, restart = False, verbose = False):
        """Initialize a model.

        It creates network and initialize agent properties in correct order.

        Params:
            restart: boolean or a pair of ints, optional.
            If True, the model is initialized from the last checkpoint,
            if a pair of ints, the model is initialized at that step from that
            milestone,
            e.g (2,0) would be the first milestone at step 2
            and (2,1) would be the second milestone at step 2.
            Default False.
        """
        self.inputs = None
        if isinstance(restart, bool):
            if restart:
                logger.info(f'Loading model state from checkpoint: {self.model_dir}')
                self.inputs = _load_model(self.model_dir)
        elif isinstance(restart, tuple):
            milestone_dir = None
            if restart[1] == 0:
                milestone_dir = f'{self.model_dir}/milestone_{restart[0]}'
            else:
                milestone_dir = f'{self.model_dir}/milestone_{restart[0]}_{restart[1]}'
            logger.info(f'Loading model state from milestone: {milestone_dir}')
            self.inputs = _load_model(milestone_dir)

        if self.inputs:
            self.graph = copy.deepcopy(self.inputs["graph"])
            self.generator_state = self.inputs["generator_state"]
            generator.set_state(self.generator_state)
            self.step_count = self.inputs["step_count"]
        else:
            torch.manual_seed(self.config.seed)
            if verbose:
                print (f"Model torch seed set to {self.config.seed}")

        self.create_network(verbose)
        if self.config.spatial:
            self.create_grid()
            self.place_agents()

        self.initialize_global_properties()
        self.initialize_agent_properties()
        self.graph = self.graph.to(self.config.device)

        if verbose:
            print(f'{self.graph.number_of_nodes()} agents initialized on {self.graph.device} device')

        weight_update_sveir(
            self.graph,
            self.config.device,
            self.steering_parameters['proximity_decay_rate'],
            self.steering_parameters['truncation_weight']
        )

        # store random generator state
        self.generator_state = generator.get_state()

        # number of edges(links) in the network
        self.number_of_edges = self.graph.number_of_edges()

        # Network Metrics
        self.average_degree = average_degree(self.graph)
        self.graph.ndata['degree'] = node_degree(self.graph)

    def run(self, verbose = False):
        """Run the model for each step until the step_target is reached."""
        # Save config to yaml file.
        self.save_model_parameters()

        self.step_first = self.step_count
        while self.step_count < self.config.step_target:
            self.step(verbose)

    def step(self, verbose):
        """Perform a single step of the model.

        After the step, the current state (graph, generator, step, and version)
        may be saved:

        The state can be saved with a fixed period (config.checkpoint_period) to
        keep a restore point in case of a crash. Only the newest checkpoint is
        retained.

        The state can also be saved at specific steps (config.milestones) to
        store specific (important) states. For example, specific states can be
        stored to start multiple runs from the same state with different
        parameters going forward. All milstones are retained. The first
        milestone at each time step X is stored in the subdirectory
        `./milestone_X`; any subsequent milestones at the same time step X are
        stored in the subdirectory `./milestone_X_i` (where i is the instance).
        """
        try:
            if verbose:
                print(f'performing step {self.step_count} of {self.config.step_target}')
            sveir_step(
                self.graph,
                self.config.device,
                self.step_count,
                self.steering_parameters,
                self.grid_environment,
                optimal_policy=self.graph.ndata["policy"]
            )
        except Exception as e:
            # TODO: Add model dump here.
            # Also check against previous save to avoid overwriting
            msg = f'Execution of step failed for step {self.step_count}'
            raise RuntimeError(msg) from e

        # save the model state every step reported by checkpoint_period and at
        # specific milestones.
        # checkpoint saves overwrite the previous checkpoint; milestone get
        # unique folders.
        # Note that milestones are not created at the first step of a run;
        # this prevents duplicate saves when running from a milestone.
        first_step = self.step_count == self.step_first
        save_checkpoint = (
            self.config.checkpoint_period > 0
            and self.step_count % self.config.checkpoint_period == 0
            )
        save_milestone = (
            self.config.milestones
            and self.step_count in self.config.milestones and not first_step
            )
        if save_checkpoint or save_milestone:
            self.inputs = {
                'graph': copy.deepcopy(self.graph),
                'generator_state': generator.get_state(),
                'step_count': self.step_count,
                'process_version': self.version
            }

            # Note that a sinlge step could be both a checkpoint and a milestone.
            # The checkpoint could be necessary to restore a crashed process while
            # the milestone is required output.
            if save_checkpoint:
                _save_model(self.model_dir, self.inputs)
            if save_milestone:
                path = f'{self.model_dir}/milestone_{self.step_count}'
                milestone_path = _make_path_unique(path)
                _save_model(milestone_path, self.inputs)

        self.step_count +=1

    def create_network(self, verbose):
        """Create intial network connecting agents.

        Makes use of intial graph type specified as model parameter.
        """
        agent_graph = network_creation(
            self.config.number_agents,
            self.config.initial_graph_type,
            verbose,
            **self.config.initial_graph_args.__dict__
            )
        self.graph = agent_graph

    def create_grid(self):
        grid_environment = grid_creation(
            **self.config.spatial_creation_args.__dict__
        )
        self.grid_environment = grid_environment

    def place_agents(self):
        self.graph.ndata['x'] = torch.zeros(self.graph.num_nodes()).float()
        self.graph.ndata['y'] = torch.zeros(self.graph.num_nodes()).float()
        grid_assignment(self.graph, self.grid_environment, **self.config.spatial_assignment_args.__dict__)

    def initialize_global_properties(self):
        """Initialize global properties/values of the model.

        Note: Global properties are initialized as tensors of length 
        corresponding to number of steps; global_theta is recorded in 
        the config file.
        """
        # Record params in config yaml file.
        self.save_model_parameters(overwrite=True)

    def initialize_agent_properties(self):
        """Initialize and assign agent properties.

        Note: agents are represented as nodes of the model graph.
        Values are initialized as tensors of length corresponding to number of
        agents, with values subsequently being assigned to the nodes.
        """
        agent_properties = {}
        agent_properties["num_infections"] = self._initialize_agent_num_infections()
        agent_properties["compartments"] = self._initialize_agents_compartment()
        agent_properties["exposure_time"] = self._initialize_agents_exposure_time()
        agent_properties["time_use"] = self._initialize_agents_time_use()
        agent_properties["home_location"] = self._initialize_agents_home_location()

        # find nearest school, water collection, place of worship
        agent_properties["school_location"] = self._find_nearest_locations(agent_properties["home_location"], "school")
        agent_properties["worship_location"] = self._find_nearest_locations(agent_properties["home_location"], "place_of_worship")
        agent_properties["water_location"] = self._find_nearest_locations(agent_properties["home_location"], "water")
        
        agent_properties["activity_choice"] = self._initialize_agents_activity_choice()
        agent_properties["wealth"] = self._initialize_agents_wealth(min=1, max=100)
        agent_properties["health"] = self._initialize_agents_health(min=1, max=100)

        # CPT and utility
        agent_properties["eta"] = self._initialize_agents_eta(min=0.5, max=1.0)
        agent_properties["omega"] = self._initialize_agents_omega(min=1.0, max=4.0)
        agent_properties["gamma"] = self._initialize_agents_gamma(min=0.2, max=0.8)
        agent_properties["alpha"] = self._initialize_agents_alpha()

        # Assign properties before calculating policy, as policy depends on them
        if isinstance(self.graph, dgl.DGLGraph):
            for key, value in agent_properties.items():
                self.graph.ndata[key] = value
        else:
            raise RuntimeError(
                'model graph must be a defined as DGLgraph object. '
                'Consider running `create_network` before initializing '
                'agent properties.'
            )

        # Now initialize policy, which depends on other agent properties
        self.graph.ndata["policy"] = self._initialize_agents_policy()


    def _initialize_agent_num_infections(self):
        tensor = torch.zeros(self.graph.num_nodes(), dtype=torch.int)
        return tensor

    def _initialize_agents_compartment(self):
        proportion = self.steering_parameters["initial_infected_proportion"]
        if not 0 <= proportion <= 1.0: 
            raise ValueError("Initial infected proportion must be between 0 and 1.")
        num_infected = round(self.graph.num_nodes() * proportion)
        tensor = torch.zeros(self.graph.num_nodes(), dtype=torch.int)
        indices = torch.randperm(self.graph.num_nodes())[:num_infected]
        tensor[indices] = 3
        return tensor
    
    def _initialize_agents_exposure_time(self):
        tensor = torch.zeros(self.graph.num_nodes(), dtype=torch.int)
        return tensor

    def _initialize_agents_time_use(self):
        tensor = torch.rand(self.graph.num_nodes(), 5) # categories: home, school, religious, social, water
        tensor /= tensor.sum(dim=1, keepdim=True)
        return tensor
    
    def _initialize_agents_home_location(self):
        tensor = torch.zeros((self.graph.num_nodes(), 2), dtype=torch.int)
        tensor[:,0] = self.graph.ndata["y"]
        tensor[:,1] = self.graph.ndata["x"]
        return tensor.float()

    def _find_nearest_locations(self, home_locations, property_name):
        """
        Finds the nearest grid locations with a given property for each agent.

        Args:
            home_locations (torch.Tensor): A tensor of agents' home coordinates.
            property_name (str): The name of the property to look for in the grid.

        Returns:
            torch.Tensor: A tensor of coordinates for the nearest location for each agent.
        """
        property_grid = self.grid_environment.grid_tensor[:, :, self.grid_environment.property_to_index[property_name]]
        property_locations = torch.stack(torch.where(property_grid == 1)).T.float()
        
        if property_locations.shape[0] == 0:
            # Handle case where no such locations exist, e.g., return home locations
            logger.warning(f"No grid locations found for property '{property_name}'. Agents will default to their home location.")
            return home_locations

        distances = torch.cdist(home_locations, property_locations)
        nearest_indices = torch.argmin(distances, dim=1)
        return property_locations[nearest_indices]

    def _initialize_agents_activity_choice(self):
        tensor = torch.zeros(self.graph.num_nodes(), dtype=torch.int)
        return tensor

    def _initialize_agents_wealth(self, min, max):
        tensor = torch.randint(min, max+1, (self.graph.num_nodes(),), dtype=torch.int)
        return tensor
    
    def _initialize_agents_health(self, min, max):
        tensor = torch.randint(min, max+1, (self.graph.num_nodes(),), dtype=torch.int)
        return tensor
    
    def _initialize_agents_eta(self, min=0.5, max=1.0):
        tensor = torch.FloatTensor(self.graph.num_nodes(),).uniform_(min, max)
        return tensor
    
    def _initialize_agents_omega(self, min=1.0, max=4.0):
        tensor = torch.FloatTensor(self.graph.num_nodes(),).uniform_(min, max)
        return tensor
    
    def _initialize_agents_gamma(self, min=0.2, max=0.8):
        tensor = torch.FloatTensor(self.graph.num_nodes(),).uniform_(min, max)
        return tensor
    
    def _initialize_agents_alpha(self):
        tensor = torch.rand(self.graph.num_nodes(),)
        return tensor
    
    def _initialize_agents_policy(self):
        """
        Initializes the health investment policy for each agent.
        
        # PERFORMANCE WARNING:
        # This function iterates through every agent and calls `value_iteration`,
        # which is a computationally expensive, iterative algorithm. This will be
        # extremely slow for a large number of agents and is a major bottleneck
        # during model initialization.
        #
        # Potential Solutions:
        # 1. Parallelization: Use Python's `multiprocessing` library to run
        #    `value_iteration` for multiple agents in parallel.
        # 2. Policy Approximation: If agents have similar parameters (`alpha`, `gamma`, etc.),
        #    group them and compute the policy only once per group.
        # 3. Pre-computation: If the parameter space is limited, pre-compute policies
        #    and load them instead of calculating them at runtime.
        """
        policies = []
        num_agents = self.graph.num_nodes()
        logger.info(f"Starting policy generation for {num_agents} agents.")
        
        for i in range(num_agents):
            print("Computing policy for agent", i)
            policies.append(
                value_iteration(
                    100,
                    self.graph.ndata["alpha"][i],
                    self.graph.ndata["gamma"][i],
                    self.steering_parameters["theta"],
                    self.graph.ndata["omega"][i],
                    self.graph.ndata["eta"][i],
                    self.steering_parameters["beta"],
                    self.steering_parameters["P_H_increase"],
                    self.steering_parameters["wealth_update_A"],
                    self.steering_parameters["P_H_decrease"]
                )
            )
        logger.info("Finished policy generation.")
        return torch.tensor(np.stack(policies))

def _make_path_unique(path, extension = ''):
    """Check whether a path already exists and make it unique if it does.

    Paths are made unique by adding "_x" to the path,
    where x is the lowest positive integer for which the path does not exist.

    Params:
        path: the path to make unique
        extension: str, optional, this extension is added to the path
          after any integer added to make the path unique.
          Note that for true extensions, this should start with a dot, e.g ".yaml"
    Returns:
        the modified path, which does not currently exist.
    """
    if Path(f'{path}{extension}').exists():
        instance = 1
        def add_instance(path, instance, extension):
            return f'{path}_{instance}{extension}'
        while Path(add_instance(path, instance, extension)).exists():
            instance += 1
        path = add_instance(path, instance, extension)
    else:
        path = path + extension
    return path

def _save_model(path, inputs):
    """Save the graph, generator_state and process_version in files."""
    Path(path).mkdir(parents=True, exist_ok=True)

    # save the graph with a label
    graph_labels = {'step_count': torch.tensor([inputs["step_count"]])}
    save_graphs(str(Path(path) / "graph.bin"), inputs["graph"], graph_labels)

    # save the generator_state
    with open(Path(path) / "generator_state.bin", 'wb') as file:
        pickle.dump([inputs["generator_state"], inputs["step_count"]], file)

    # save the process version
    with open(Path(path) / "process_version.md", 'w') as file:
        file.writelines(
            [inputs["process_version"] + '\n',
            f'step={inputs["step_count"]}\n']
            )


def _load_model(path):
    # Load model graph
    path_graph = Path(path) / "graph.bin"
    if not path_graph.is_file():
        raise ValueError(f'The path {path_graph} is not a file.')

    graph, graph_labels = load_graphs(str(path_graph))
    graph = graph[0]
    graph_step = graph_labels['step_count'].tolist()[0]

    # Load generator_state
    path_generator_state = Path(path) / "generator_state.bin"
    if not path_generator_state.is_file():
        raise ValueError(f'The path {path_generator_state} is not a file.')

    with open(path_generator_state, 'rb') as file:
        generator, generator_step = pickle.load(file)

    # Load process version
    path_process_version = Path(path) / "process_version.md"
    if not path_process_version.is_file():
        raise ValueError(f'The path {path_process_version} is not a file.')

    with open(path_process_version) as file:
        process_version = file.readlines()[0]

    # Check if graph_step, generator_step and data_step are the same
    if graph_step != generator_step:
        msg = 'The step count in the graph and generator_state are not the same.'
        raise ValueError(msg)

    # Check if the saved version and current process version are the same
    version_path = Path(__file__).resolve().parents[2] / 'version.md'
    current_version = version_path.read_text().splitlines()[0]
    if process_version != current_version:
        logger.warning(
            'Warning: loading model generated using earlier process version: '
            f'{process_version}.'
            )

    # Show which step is loaded
    logger.warning(f'Loading model state from step {generator_step}.')

    inputs = {
        'graph': graph,
        'generator_state': generator,
        'step_count': generator_step,
        'process_version': process_version
    }
    return inputs