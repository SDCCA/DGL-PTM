import dgl
import networkx as nx
import torch

from dgl_ptm.model.base_model import Model
from dgl_ptm.network.network_creation import network_creation

def sample_distribution_tensor(type, distParameters, nSamples, round=False, decimals=None):
    """
    create and return samples from different distributions

    :param type: Type of distribution to sample
    :param distParameters: array of parameters as required/supported by requested distribution type
    :param nSamples: number of samples to return (as 1d tensor)
    :param round: optional, whether the samples are to be rounded
    :param decimals: optional, required if round is specified. decimla olaces to round to
    """
    if type == 'uniform':
        dist = torch.distributions.uniform.Uniform(torch.tensor(distParameters[0]),torch.tensor(distParameters[1])).sample(torch.tensor([nSamples]))
    elif type == 'normal':
        dist = torch.distributions.normal.Normal(torch.tensor(distParameters[0]),torch.tensor(distParameters[1])).sample(torch.tensor([nSamples]))
    elif type == 'bernoulli':
        dist = torch.distributions.bernoulli.Bernoulli(probs=distParameters[0],logits=distParameters[1],validate_args=None).sample(torch.tensor([nSamples]))
    else:
        raise NotImplementedError('Currently only uniform, normal and bernoulli distributions are supported')

    if round:
        if decimals == None:
            raise ValueError('rounding requires decimals of rounding accuracy to be specified')
        else:
            return torch.round(dist,decimals=decimals)
    else:
        return dist


class PovertyTrapModel(Model):
    """
    Poverty Trap model as derived model class

    """

    #default values as class variable 
    default_model_parameters = {'number_agents': 100 , 
    'gamma_vals':torch.tensor([0.3,0.45]) , 
    'sigma': torch.tensor(0.5), 
    'cost_vals': torch.tensor([0.,0.45]) , 
    'tec_levels': torch.tensor([0,1]), 
    'tec_dist': {'type':'bernoulli','parameters':[0.5,None],'round':False,'decimals':None}, 
    'capital_dist': {'type':'uniform','parameters':[0.1,10.],'round':False,'decimals':None}, 
    'alpha_dist': {'type':'normal','parameters':[1.08,0.074],'round':False,'decimals':None},
    'lam_dist': {'type':'uniform','parameters':[0.1,0.9],'round':True,'decimals':1},
    'initial_graph_type': 'barabasi-albert',
    'number_of_steps':100,
    'restart_step':0,
    '_step_counter':0}

    def __init__(self,*, model_identifier=None, restart=False, savestate=None):
        """
        restore from a savestate (TODO) or create a PVT model instance.
        Checks whether a model indentifier has been specified.
        """
        self.restart = restart
        if restart:
            if savestate==None:
                raise ValueError('When restarting a simulation an intial savestate must be supplied')
            else:
                self.restart = restart
                #TODO implement restart
                pass
        else:
            super().__init__(model_identifier = model_identifier)
            if self._model_identifier == None:
                raise ValueError('A model identifier must be specified')
            self.number_agents = None
            self.gamma_vals = None
            self.sigma = None
            self.cost_vals = None
            self.tec_levels = None
            self.tec_dist = None
            self.capital_dist = None
            self.alpha_dist = None
            self.lam_dist = None 
            self.initial_graph_type = None
            self.model_graph = None
            self.number_of_steps = None
            self.restart_step = None
            self._step_counter = None


    def set_model_parameters(self,*,parameterFilePath=None, default=True, **kwargs):
        """
        Load (TODO) or set model parameters

        :param parameterFlePath: optional, path to restore file (TODO)
        :param default: Specify whether default values should be used (True;default)
        :param **kwargs: flexible passing of mode parameters. Only those supported by the model are accpted.
                         If parameters are passed, non-specifed parameters will be set with defaults.

        """
        if parameterFilePath != None:
            pass
            #TODO restore from file incl. check for model identifer
        else:
            modelpars = self.__dict__.keys()
            if default:
                for modelpar in modelpars:
                    if modelpar not in ['_model_identifier','model_graph']:
                        self.__dict__[modelpar] = self.default_model_parameters[modelpar]
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

    def initialize_model(self):
        self.create_network()
        self.initialize_agent_properties()

    def step(self,step_num,stepnum_resume=None):
        """
        TODO
        """
        pass

    def run(self):
        goal_step_count = self.number_of_steps
        if self.restart:
            step_counter = self.restart_step
        else:
            step_counter = 0    
        while step_counter < goal_step_count:
            self._step_counter = step_counter
            self.step()
            step_counter += 1




    def create_network(self):
        """
        Create intial network connecting agents. Makes use of intial graph type specified as model parameter
        """

        agent_graph = network_creation(self.number_agents, self.initial_graph_type)
        self.model_graph = agent_graph

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
        agentsTecLevel, agentsGamma, agentsCost = self._initialize_agents_tec()

        if isinstance(self.model_graph,dgl.DGLgraph):
            self.model_graph.ndata['k'] = agentsCapital
            self.model_graph.ndata['alpha'] = agentsAlpha
            self.model_graph.ndata['lambda'] = agentsLam
            self.model_graph.ndata['sigma'] = agentsSigma
            self.model_graph.ndata['tec'] = agentsTecLevel
            self.model_graph.ndata['gamma'] = agentsGamma
            self.model_graph.ndata['cost'] = agentsCost
        else:
            raise RuntimeError('model graph must be a defined DGLgraph object. Consder running `create_network` before initializing agent properties')

    def _initialize_agents_capital(self):
        """
        Initialize agents captial as a 1d tensor sampled from the specified intial capita distribution
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
        agentsLam = sample_distribution_tensor(self.lam_dist['type'],self.lam_dist['parameters'],self.number_agents,round=self.lam_dist['round'],decimals=self.lam_dist['decimals'])
        return agentsLam

    def _initialize_agents_sigma(self):
        """
        Initialize agents sigma as a 1d tensor 
        """
        agentsSigma = torch.ones(self.number_agents)*self.sigma
        return agentsSigma

    def _initialize_agents_tec(self):
        """
        Initialize the agents technology level distribution as 1d tensor sampled from the specified intial technology level distribution.
        Initialize agents gamma and cost distributions according to their tec level and the spefied initial gamma and cost
        values associated with that tech level
        """
        agentsTecLevel = sample_distribution_tensor(self.tec_dist['type'],self.tec_dist['parameters'],self.number_agents,round=self.tec_dist['round'],decimals=self.tec_dist['decimals'])
        agentsGamma = torch.zeros(self.number_agents)
        agentsCost = torch.zeros(self.number_agents)
        for i in range(len(self.tec_levels)):
            tec_mask = agentsTecLevel == i
            agentsGamma[tec_mask] = self.gamma_vals[i]
            agentsCost[tec_mask] = self.cost_vals[i]   
        return agentsTecLevel, agentsGamma, agentsCost
