import xarray as xr 
from pathlib import Path
import os
import dgl
import dask.array as da
from os.path import exists

def data_collection(agent_graph, timestep, npath='./agent_data', epath='./edge_data', ndata = ['all'], edata = ['all'], format = 'xarray', mode = 'w-'):
    ''' 
        data_collection - collects data from agents and edges.

        Args:
            agent_graph: DGLGraph with agent nodes and edges connecting agents.
            timestep = current timestep to name folder for edge properties
            npath = path to store node data.
            epath = path to store edge data with one file for each timestep.
            ndata = node data properties to be stored. 
                ['all'] implies all node properties will be saved
            edata = edge data properties to be stored. 
                ['all'] implies all edge properties will be saved
            format = storage format
                ['xarray'] saves the properties in zarr format with xarray dataset
            mode = zarr write mode.

        Output:

    '''
    if ndata == ['all']:
        ndata = list(agent_graph.node_attr_schemes().keys())    
    if ndata[0] == 'all_except':
        ndata = list(agent_graph.node_attr_schemes().keys() - ndata[1])
    if edata == ['all']:
        edata = list(agent_graph.edge_attr_schemes().keys())
    agent_graph = agent_graph.to('cpu')
    _node_property_collector(agent_graph, npath, ndata, timestep, format, mode)
    _edge_property_collector(agent_graph, epath, edata, timestep, format, mode)


def _node_property_collector(agent_graph, npath, ndata, timestep, format, mode):
    if os.environ["DGLBACKEND"] == "pytorch":
        if format == 'xarray':
            agent_data_instance = xr.Dataset()
            for prop in ndata:
                _check_nprop_in_graph(agent_graph, prop)
                agent_data_cpu=agent_graph.ndata[prop][:,None].cpu()
                agent_data_instance = agent_data_instance.assign(prop=(['n_agents','n_time'], agent_data_cpu.numpy()))
                #below code didn't work above is a fix, but issue about gpu to cpu transfer remains. also why must copies be made rather than methods called
               # agent_data_instance = agent_data_instance.assign(prop=(['n_agents','n_time'], da.from_array(agent_graph.ndata[prop][:,None].cpu().numpy())))
                agent_data_instance = agent_data_instance.rename(name_dict={'prop':prop})
            if timestep == 0:
                agent_data_instance.to_zarr(npath, mode = mode)
            else:
                agent_data_instance.to_zarr(npath, append_dim='n_time')
        else:
            raise NotImplementedError("Only 'xarray' format currrent available")
    else:
        raise NotImplementedError("Data collection currently only implemented for pytorch backend")
        

def _edge_property_collector(agent_graph, epath, edata, timestep, format, mode): 
    if os.environ["DGLBACKEND"] == "pytorch":
        if format == 'xarray':
            edge_data_instance = xr.Dataset(coords=dict(
                                            source=(["n_edges"], agent_graph.edges()[0].cpu()),
                                            dest=(["n_edges"], agent_graph.edges()[1].cpu()),
                                            ))
            for prop in edata:
                _check_eprop_in_graph(agent_graph, prop)
                edge_data_cpu=agent_graph.edata[prop][:,None].cpu()
                edge_data_instance = edge_data_instance.assign(property=(['n_edges','time'], edge_data_cpu.numpy()))
         #same issues as above for node collection
                #edge_data_instance = edge_data_instance.assign(property=(['n_edges','time'], da.from_array(agent_graph.edata[prop][:,None].cpu().numpy())))

                edge_data_instance = edge_data_instance.rename_vars(name_dict={'property':prop})
            edge_data_instance.to_zarr(Path(epath)/(str(timestep)+'.zarr'), mode = mode)
        else:
            raise NotImplementedError("Only 'xarray' mode currrent available")
    else:
        raise NotImplementedError("Data collection currently only implemented for pytorch backend")
    
def _check_nprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.node_attr_schemes().keys():
        raise ValueError(f"{prop} is not a node property. Please choose from {agent_graph.node_attr_schemes().keys()}")
    
def _check_eprop_in_graph(agent_graph, prop):
    if prop not in agent_graph.edge_attr_schemes().keys():
        raise ValueError(f"{prop} is not an edge property. Please choose from {agent_graph.edge_attr_schemes().keys()}")
