import torch
import .nn_arch as nn_arch


def load_model(model_path,device):
    '''Load a model from a particular .pth file and assemble using structure contained in nn_arch.py'''    

    modelinfo = torch.load(model_path, map_location=device)

    config = modelinfo['config']

    model = getattr(nn_arch,config['arch']['type'])(**config['arch']['args'])

    model.load_state_dict(modelinfo['state_dict'])

    if config['data_loader']['cons_scale']==True:
        scale=config['data_loader']['scale']
    else:
        scale=1

    return model, scale
