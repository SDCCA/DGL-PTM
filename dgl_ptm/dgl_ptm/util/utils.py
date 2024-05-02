import torch
import dgl_ptm.util.nn_arch.nn_arch as nn_arch
import parse_config




def load_consumption_model(model_path,device):
    '''Load a model from a particular .pth file and assemble using structure contained in nn_arch.py'''
    print("entered load_consumption_model")

    model_path="/Users/victoria/Documents/Scripts/Python/DGL-PTM/DGL_testing/nn_data/both_PudgeFiveLayer_1024/0409_175055/model_best.pth"
    modelinfo = torch.load(model_path, map_location=torch.device('cpu'))


    print("loaded info")

    config = modelinfo['config']

    model = getattr(nn_arch,config['arch']['type'])(**config['arch']['args'])

    model.load_state_dict(modelinfo['state_dict'])

    print("loaded state dict")


    if config['data_loader']['cons_scale']==True:
        scale=config['data_loader']['scale']
    else:
        scale=1

    return model, scale
