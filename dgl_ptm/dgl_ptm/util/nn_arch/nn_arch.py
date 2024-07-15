import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from abc import abstractmethod


class BaseModel(nn.Module):
    """
    Base class for all models
    """
    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

class MnistModel(BaseModel):
    def __init__(self, num_classes=10):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, num_classes)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)



class ThreeLayer(BaseModel):
    def __init__(self, input_dim=4, output_dim=2, max_nodes=256):
        super(ThreeLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        # Define the layers
        self.layer1 = nn.Linear(input_dim, max_nodes)
        self.layer2 = nn.Linear(int(max_nodes), int(max_nodes/2))
        self.layer3 = nn.Linear(int(max_nodes/2), output_dim)
        
        # Initialize the weights
        self._init_weights()
        
    def _init_weights(self):
        # Custom weight initialization if needed
        pass
    
    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x



class FourLayer(BaseModel):
    def __init__(self, input_dim=4, output_dim=2, max_nodes=512):
        super(FourLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        # Define the layers
        self.layer1 = nn.Linear(input_dim, max_nodes)
        self.layer2 = nn.Linear(int(max_nodes), int(max_nodes/2))
        self.layer3 = nn.Linear(int(max_nodes/2), int(max_nodes/4))
        self.layer4 = nn.Linear(int(max_nodes/4), output_dim)
        
        # Initialize the weights
        self._init_weights()
        
    def _init_weights(self):
        # Custom weight initialization if needed
        pass
    
    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = self.layer4(x)
        return x


class FiveLayer(BaseModel):
    def __init__(self, input_dim=4, output_dim=2, max_nodes=512):
        super(FiveLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        # Define the layers
        self.layer1 = nn.Linear(input_dim, max_nodes)
        self.layer2 = nn.Linear(int(max_nodes), int(max_nodes/2))
        self.layer3 = nn.Linear(int(max_nodes/2), int(max_nodes/4))
        self.layer4 = nn.Linear(int(max_nodes/4), int(max_nodes/8))
        self.layer5 = nn.Linear(int(max_nodes/8), output_dim)
        
        # Initialize the weights
        self._init_weights()
        
    def _init_weights(self):
        # Custom weight initialization if needed
        pass
    
    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x



class PudgeFiveLayer(BaseModel):
    def __init__(self, input_dim=4, output_dim=2, max_nodes=512):
        super(PudgeFiveLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        # Define the layers
        self.layer1 = nn.Linear(input_dim, int(max_nodes/2))
        self.layer2 = nn.Linear(int(max_nodes/2),max_nodes)
        self.layer3 = nn.Linear(max_nodes, int(max_nodes/2))
        self.layer4 = nn.Linear(int(max_nodes/2), int(max_nodes/4))
        self.layer5 = nn.Linear(int(max_nodes/4), output_dim)
        
        # Initialize the weights
        self._init_weights()
        
    def _init_weights(self):
        # Custom weight initialization if needed
        pass
    
    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = self.layer5(x)
        return x





class PudgeSixLayer(BaseModel):
    def __init__(self, input_dim=4, output_dim=2, max_nodes=512):
        super(PudgeSixLayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.max_nodes = max_nodes
        # Define the layers
        self.layer1 = nn.Linear(input_dim, int(max_nodes/4))
        self.layer2 = nn.Linear(int(max_nodes/4), int(max_nodes/2))
        self.layer3 = nn.Linear(int(max_nodes/2),max_nodes)
        self.layer4 = nn.Linear(max_nodes, int(max_nodes/2))
        self.layer5 = nn.Linear(int(max_nodes/2), int(max_nodes/4))
        self.layer6 = nn.Linear(int(max_nodes/4), output_dim)
        
        # Initialize the weights
        self._init_weights()
        
    def _init_weights(self):
        # Custom weight initialization if needed
        pass
    
    def forward(self, x):
        # Forward pass through the layers
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = F.relu(self.layer3(x))
        x = F.relu(self.layer4(x))
        x = F.relu(self.layer5(x))
        x = self.layer6(x)
        return x