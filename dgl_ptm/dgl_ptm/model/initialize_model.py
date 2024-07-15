import dgl
import networkx as nx
import torch
import yaml

from dgl_ptm.network.network_creation import network_creation
from dgl_ptm.model.step import ptm_step
from dgl_ptm.agentInteraction.weight_update import weight_update


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
    elif type == 'truncnorm':
        # distParameters are mean, standard deviation, min, and max. cdf(x)=(1+erf(x/2^0.5))/2. cdf^-1(x)=2^0.5*erfinv(2*x-1).
        trunc_val_min = (distParameters[2]-distParameters[0])/distParameters[1]
        trunc_val_max = (distParameters-distParameters[0])/distParameters[1]
        cdf_min = (1 + torch.erf(trunc_val_min / torch.sqrt(torch.tensor(2.0))))/2
        cdf_max = (1 + torch.erf(trunc_val_max / torch.sqrt(torch.tensor(2.0))))/2

        uniform_samples = torch.rand(size)
        sample_ppf = torch.sqrt(torch.tensor(2.0)) * torch.erfinv(2 *(cdf_min + (cdf_max - cdf_min) * uniform_samples) - 1)

        dist = destParameters[0] + destParameters[1] * sample_ppf
    

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
    'seed':0,
    'gamma_vals':torch.tensor([0.3,0.45]) , #for pseudo income
    'sigma_dist': {'type':'uniform','parameters':[0.05,1.94],'round':True,'decimals':1},
    'cost_vals': torch.tensor([0.,0.45]), #for pseudo income
    'tec_levels': torch.tensor([0,1]), #check if deletable
    'a_theta_dist': {'type':'uniform','parameters':[0.1,1],'round':False,'decimals':None},
    'sensitivity_dist':{'type':'uniform','parameters':[0.0,1],'round':False,'decimals':None},
    'tec_dist': {'type':'bernoulli','parameters':[0.5,None],'round':False,'decimals':None}, 
    'capital_dist': {'type':'uniform','parameters':[0.1,10.],'round':False,'decimals':None}, 
    'alpha_dist': {'type':'normal','parameters':[1.08,0.074],'round':False,'decimals':None},
    'lam_dist': {'type':'uniform','parameters':[0.05,0.94],'round':True,'decimals':1},
    'initial_graph_type': 'barabasi-albert',
    'initial_graph_args': {'seed': 0, 'new_node_edges':1},
    'device': 'cpu',
    'step_count':0,
    'step_target':20,
    'steering_parameters':{'npath':'./agent_data.zarr',
                            'epath':'./edge_data', 
                            'ndata':['all_except',['a_table']],
                            'edata':['all'],
                            'mode':'xarray',
                            'wealth_method':'singular_transfer',
                            'income_method':'pseudo_income_generation',
                            'tech_gamma': torch.tensor([0.3,0.35,0.45]),
                            'tech_cost': torch.tensor([0,0.15,0.65]),
                            'consume_method':'pseudo_consumption',
                            'nn_path': None,
                            'adapt_m':torch.tensor([0,0.5,0.9]),
                            'adapt_cost':torch.tensor([0,0.25,0.45]),
                            'depreciation': 0.6,
                            'discount': 0.95,
                            'm_theta_dist': {'type':'multinomial','parameters':[[0.02 ,0.03, 0.05, 0.9],[0.7, 0.8, 0.9, 1]],'round':False,'decimals':None},
                            'm_attach_dist': {'type':'uniform','parameters':[0.001,1],'round':False,'decimals':None},
                            'del_method':'probability',
                            'del_threshold':0.05,
                            'ratio':0.1,
                            'weight_a':0.69,
                            'weight_b':35, 
                            'truncation_weight':1.0e-10,
                            'step_type':'default'}}

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
            self.number_agents = None
            self.seed = None
            self.gamma_vals = None
            self.sigma_dist = None
            self.cost_vals = None
            self.tec_levels = None
            self.tec_dist = None
            self.a_theta_dist = None
            self.sensitivity_dist = None
            self.capital_dist = None
            self.alpha_dist = None
            self.lam_dist = None 
            self.initial_graph_type = None
            self.initial_graph_args = None
            self.model_graph = None
            self.device = None
            self.step_count = None
            self.step_target = None
            self.steering_parameters = None

    def set_model_parameters(self,*,parameterFilePath=None, default=True, **kwargs):
        """
        Load or set model parameters

        :param parameterFlePath: optional, path to parameter file
        :param default: Specify whether default values should be used (True;default)
        :param **kwargs: flexible passing of mode parameters. Only those supported by the model are accepted.
                         If parameters are passed, non-specifed parameters will be set with defaults.

        """
        modelpars = self.__dict__.keys()
        if parameterFilePath != None:
            with open(parameterFilePath, 'r') as readfile:
                try:
                    self.__dict__ = yaml.safe_load(readfile)
                except yaml.YAMLError as exc:
                    raise SyntaxError(exc)
                
            for modelpar in modelpars:
                if modelpar not in ['_model_identifier','model_graph']:
                    if type(self.__dict__[modelpar]) is list:
                        self.__dict__[modelpar] = torch.tensor(self.__dict__[modelpar])
        else:
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
        """
        convenience fucntion to create network and initiliize agent properties in correct order, thereby initializing a model
        """
        torch.manual_seed(self.seed)
        self.create_network()
        self.initialize_agent_properties()
        self.initialize_model_properties()
        weight_update(self.model_graph, self.device, self.steering_parameters['weight_a'], self.steering_parameters['weight_b'], self.steering_parameters['truncation_weight'])

    def create_network(self):
        """
        Create intial network connecting agents. Makes use of intial graph type specified as model parameter
        """

        agent_graph = network_creation(self.number_agents, self.initial_graph_type, **self.initial_graph_args)
        self.model_graph = agent_graph.to(self.device)
        print(f'Created graph with {self.model_graph.number_of_nodes()} agents.')
        print(f"At initialization, graph has {torch.sum(agent_graph.edges(order='eid')[0]==agent_graph.edges(order='eid')[1])} auto edges and {agent_graph.edges(order='eid')[0].size(0)-torch.unique(torch.stack((agent_graph.edges(order='eid')[0],agent_graph.edges(order='eid')[1]), dim=1),dim=0).size(0)} duplicate edges.")

    def initialize_model_properties(self):
        """
        Initialize model properties.
        Values are initialized as tensors of length corresponding to number of time steps.
        """
        modelTheta = self._initialize_model_theta()
        self.steering_parameters['modelTheta'] = modelTheta
        attachProb = self._initialize_attach_prob()
        self.steering_parameters['attachProb'] = attachProb

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

        if isinstance(self.model_graph,dgl.DGLGraph):
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
            raise RuntimeError('model graph must be a defined DGLgraph object. Consder running `create_network` before initializing agent properties')


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
        agentsLam = sample_distribution_tensor(self.lam_dist['type'],self.lam_dist['parameters'],self.number_agents,round=self.lam_dist['round'],decimals=self.lam_dist['decimals'])
        return agentsLam

    def _initialize_agents_sigma(self):
        """
        Initialize agent sigma as a 1d tensor 
        """
        agentsSigma = sample_distribution_tensor(self.sigma_dist['type'],self.sigma_dist['parameters'],self.number_agents,round=self.sigma_dist['round'],decimals=self.sigma_dist['decimals'])
        return agentsSigma

    def _initialize_agents_tec(self):
        """
        Initialize the agent technology level distribution as 1d tensor sampled from the specified initial technology level distribution.
        Initialize agent gamma and cost distributions according to tec level and the spefied initial gamma and cost
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

    def step(self):
        try:
            ptm_step(self.model_graph, self.device, self.step_count,self.steering_parameters)
            self.step_count +=1
        except:
            #TODO add model dump here. Also check against previous save to avoid overwriting


            raise RuntimeError(f'execution of step failed for step {self.step_count}')


    def run(self):
        while self.step_count < self.step_target:
            print(f'performing step {self.step_count+1} of {self.step_target}')
            self.step()
