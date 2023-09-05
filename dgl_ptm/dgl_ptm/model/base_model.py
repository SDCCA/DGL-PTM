class Model(object):
    """
    Abstract model class
    """

    def __init__(self,model_identifier=None):
        self._model_identifier = model_identifier
        self.number_agents = None
    
    def set_model_parameters(self):
        raise NotImplementedError('setting model parameters is not implemented for this class.')
    
    def initialize_model(self):
        raise NotImplementedError('model initializaton is not implemented for this class.')
    
    def run(self):
        raise NotImplementedError('method ".run()" is not implemented for this class.')
    
    def step(self):
        raise NotImplementedError('method ".step()" is not implemented for this class.')