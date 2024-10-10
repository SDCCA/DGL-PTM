import torch
import dgl_ptm.util.nn_arch.nn_arch as nn_arch


import os


def load_consumption_model(nn_path,device):
    '''Load a model from a particular .pth file and assemble using structure contained in nn_arch.py'''
    #print("entered load_consumption_model")
    nn_path=f'{os.getcwd()}{nn_path}'
    modelinfo = torch.load(nn_path, map_location=torch.device(device))


    #print("loaded info")

    config = modelinfo['config']

    model = getattr(nn_arch,config['arch']['type'])(**config['arch']['args'])

    model.load_state_dict(modelinfo['state_dict'])

    #print("loaded state dict")


    if "cons_scale" in config["data_loader"]["args"]:
        cons_scale=config['data_loader']['args']['i_a_scale']
    else:
        cons_scale=1    
    if "i_a_scale" in config["data_loader"]["args"]:
        i_a_scale=config['data_loader']['args']['i_a_scale']
    else:
        i_a_scale=1

    return model, cons_scale, i_a_scale
