import dgl_ptm
import torch
import os
import time
os.environ["DGLBACKEND"] = "pytorch"

# Test pytorch cuda is available
if not torch.cuda.is_available():
    raise SystemError('GPU access not available with PyTorch') 

nagents = [50000]
runtime = []
for i in nagents:
    start = time.time()
    # Create an instance of the model class
    model = dgl_ptm.PovertyTrapModel(model_identifier=f'test{i}')

    # Set the model and simulation parameters
    # Using DEFAULT values
    model.set_model_parameters(**{"device": "cuda"})
    model.number_agents = i
    model.step_target = 100
    model.steering_parameters['mode'] = 'w'
    # OR
    # using a CONFIG file
    # model.set_model_parameters(default=True, parameterFilePath='../dgl_ptm/dgl_ptm/config.yaml')

    # Initialize the model
    model.initialize_model()

    # Run the model
    model.run()
    end = time.time()
    runtime.append(end-start)    

for i in range(len(nagents)):
    print ('Time taken to run ', nagents[i],' agents for 100 timesteps is ', runtime[i])
