import torch
import dgl_ptm.util.nn_arch.nn_arch as nn_arch




def load_consumption_model(nn_path,device):
    '''Load a model from a particular .pth file and assemble using structure contained in nn_arch.py'''
    #print("entered load_consumption_model")

    modelinfo = torch.load(nn_path, map_location=torch.device(device))


    #print("loaded info")

    config = modelinfo['config']

    model = getattr(nn_arch,config['arch']['type'])(**config['arch']['args'])

    model.load_state_dict(modelinfo['state_dict'])

    #print("loaded state dict")


    if config['data_loader']['args']['cons_scale']==True:
        scale=config['data_loader']['args']['scale']
    else:
        scale=1

    return model, scale
