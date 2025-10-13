"""This module provides a function for agent movement.

Functions:
- move_agents: Moves agents in the grid environment
- _identify_moving_agents: Identifies which agents will move based on model parameters
- _select_agent_positions: Selects new positions for agents based on model parameters
- _set_agent_positions: Writes new positions for agents in the model graph
- _random_jump: Randomly selects new agent position from entire grid
- _square: Constructs mask for square movement pattern
- _circle: Constructs mask for circular movement pattern
- _radial: Constructs mask for radial movement pattern
- _cross: Constructs mask for cross movement pattern
"""

import torch
import dgl
import random
import warnings

def move_agents(model_graph, model_params, grid_environment, device, agentIDs=None, new_positions=None, agent_selection=None,movement_function=None):
    """Move agents around the grid environment.
    
    Args:
        model_graph (DGLGraph): All agent node and edge data
        grid_environment (GridEnvironment): agent environment information
        model_params (dict): model parameters
        agentIDs (torch.Tensor, optional): Specific agents to move
        new_positions (torch.Tensor, optional): New positions for the specified agents
    """

    if len(grid_environment.space) > 2:
        raise NotImplementedError('At this time, only 2D movement is supported. 3D movement feature is in development.')
    if agentIDs is None:
        agentIDs = _identify_moving_agents(model_graph, model_params)
    if new_positions is None:           
        if agentIDs.shape[0] > 0 and model_params['movement_function'] is not None:
            new_positions, agentIDs = _select_agent_positions(model_graph, model_params, grid_environment, agentIDs, movement_function=movement_function)
        else:
           raise NotImplementedError(f'Movement was requested, but no values for new_positions or movement_function were provided.')

    _set_agent_positions(model_graph, grid_environment, agentIDs, new_positions)

def _identify_moving_agents(model_graph, model_params):
    if 'moving_agents' not in model_params or model_params['moving_agents']in ["All","all"]:
            agentIDs = model_graph.nodes()
    else:
        if model_params['moving_agents'] == 'random':
            n = torch.randint(model_graph.num_nodes(), (1,)).item()
            agentIDs = torch.randperm(model_graph.num_nodes())[:n]           
        elif model_params['moving_agents'] == 'n_random':
            if 'n_moving_agents' not in model_params:
                raise ValueError("A value for n_moving_agents must be specified when moving_agents is set to 'n_random'.")
            n = model_params['n_moving_agents']
            if n > model_graph.num_nodes():
                raise ValueError("n_moving_agents cannot be greater than the total number of agents.")
            agentIDs = torch.randperm(model_graph.num_nodes())[:n]
        elif model_params['moving_agents'] == 'ratio_random':
            if 'ratio_moving_agents' not in model_params:
                raise ValueError("A value for ratio_moving_agents must be specified when moving_agents is set to 'ratio_random'.")
            ratio = model_params['ratio_moving_agents']
            if ratio < 0 or ratio > 1:
                raise ValueError(" The value for ratio_moving_agents must be between 0 and 1.")
            n = int(ratio * model_graph.num_nodes())
            agentIDs = torch.randperm(model_graph.num_nodes())[:n]
        elif model_params['moving_agents'] == 'variable_based':
            if 'moving_probability_variable' not in model_params:
                raise ValueError("A moving_probability_variable must be specified when moving_agents is set to 'variable_based'.")
            if model_params['moving_probability_variable'] not in model_graph.ndata.keys():
                raise ValueError("The specified moving_probability_variable is not a node feature in the model_graph.")
            thresholds = torch.rand(model_graph.num_nodes())
            variable_data = model_graph.ndata[model_params['moving_probability_variable']]
            agentIDs = (variable_data > thresholds).nonzero(as_tuple=True)[0]
        else:
            raise ValueError(f"Movement using the agent selection method {model_params['moving_agents']} is unavailable.")
    return agentIDs
            
def _select_agent_positions(model_graph, model_params, grid_environment, agentIDs, movement_function=None):
    """Select new positions for specified agents in the grid environment.
    This function returns a position for each agent in agentIDs based on the 
    movement_function model parameter. This may be as a random jump or 
    by creating a mask of possible moves and applying it to current agent  
    positions, cropping as needed to fit bounds of the grid environment; a new
    position is then selected from the possible moves at random or based on a 
    grid environment property.
    Note: If using probability-based seeking behavior, ensure that the property
    layer 1) exists in the grid environment, 2) is populated with non negatives,
    3) has resolution coarser than 1E-8 (tolerance for zero).
    
    Args:
        model_graph (DGLGraph): All agent node and edge data
        model_params (dict): model parameters
        grid_environment (GridEnvironment): agent environment information
        agentIDs (torch.Tensor): Specific agents to move
    
    Returns:
        torch.Tensor: New positions for the specified agents
    """
    positions = torch.stack((model_graph.ndata['x'][agentIDs], model_graph.ndata['y'][agentIDs]), dim=1)
    x_max = grid_environment.grid_shape[0] - 1
    y_max = grid_environment.grid_shape[1] - 1

    movement_function = movement_function if movement_function is not None else model_params.get('movement_function', None)

    if isinstance(movement_function, dict):
        pattern = movement_function['pattern']
        range_min = movement_function.get('range_min', 0)
        range_max = movement_function.get('range_max', 1)
        seeking_property = movement_function.get('seeking_property', None)
        seeking_behavior = movement_function.get('seeking_behavior', None)
    else:
        pattern = movement_function
        range_min = 0
        range_max = 1
        seeking_property = None
        seeking_behavior = None
    if range_min > range_max:
        raise ValueError(f"The value for range_min ({range_min}) cannot exceed range_max ({range_max}).")
    if range_min < 0 or range_max < 0:
        raise ValueError(f"The values for range_min ({range_min}) and range_max ({range_max}) cannot be negative.")
    if seeking_property is not None and seeking_property not in grid_environment.property_to_index.keys():
        raise ValueError(f"The selected seeking_property, {seeking_property}, does not correspond to the name of a layer in the GridEnvironment.")
    if range_min > max(grid_environment.grid_shape[0]-1, grid_environment.grid_shape[1]-1):
        raise ValueError("The range_min exceeds the GridEnvironment boundaries.")
    if range_max > max(grid_environment.grid_shape[0]-1, grid_environment.grid_shape[1]-1):
        warnings.warn("The range_max exceeds the GridEnvironment boundaries.", UserWarning)

    if pattern in ['random_jump', 'random', None]:
        new_positions = _random_jump(grid_environment, agentIDs)
        return new_positions, agentIDs

    if pattern == 'square':
        possibility_mask = _square(range_min,range_max)
    elif pattern == 'circle':
        possibility_mask = _circle(range_min,range_max)
    elif pattern == 'radial':
        possibility_mask = _radial(range_min,range_max)
    elif pattern == 'cross':
        possibility_mask = _cross(range_min,range_max)
    else:
        raise ValueError(f"A mask could not be created for the movement function {model_params['movement_function']} with range ({range_min}, {range_max}).")
    
    possibility_indices = possibility_mask.nonzero(as_tuple=False)
    if possibility_indices.shape[0] == 0:
        raise ValueError(f"The movement function {model_params['movement_function']} with range ({range_min}, {range_max}) results in no possible moves.")
    possibility_indices_xy = possibility_indices[:, [1, 0]]
    relative_dx_dy = possibility_indices_xy - range_max

    possible_positions = positions.unsqueeze(1) + relative_dx_dy.unsqueeze(0)

    bounded_possible_positions = (possible_positions[:,:,0] >= 0) & \
                                 (possible_positions[:,:,0] <= x_max) & \
                                 (possible_positions[:,:,1] >= 0) & \
                                 (possible_positions[:,:,1] <= y_max)

    valid_movement = bounded_possible_positions.any(dim=1)
    positions = positions[valid_movement]
    agentIDs = agentIDs[valid_movement]
    possible_positions = possible_positions[valid_movement]
    bounded_possible_positions = bounded_possible_positions[valid_movement]

    if seeking_property is not None:
        agent_index, position_index = torch.where(bounded_possible_positions)
        coordinates = possible_positions[agent_index, position_index]
        position_property = grid_environment[seeking_property][coordinates[:, 0], coordinates[:, 1]]

        if seeking_behavior in ['minimum','min']:
            possibility_scores = torch.full(bounded_possible_positions.shape, float('inf'))
            possibility_scores[bounded_possible_positions] = position_property
            selection=torch.argmin(possibility_scores, dim=1)
        elif seeking_behavior in ['maximum','max']:
            possibility_scores = torch.full(bounded_possible_positions.shape, float('-inf'))
            possibility_scores[bounded_possible_positions] = position_property
            selection=torch.argmax(possibility_scores, dim=1)
        elif seeking_behavior in ['probability','prob']:
            if torch.any(grid_environment[seeking_property] < 0):
                raise ValueError('The seeking_property must not have negative' \
                ' values if the probability seeking_behavior is used. ' \
                'Please choose a property layer with no negative weights.')
            possibility_scores = torch.zeros(bounded_possible_positions.shape, dtype=torch.float32)
            possibility_scores[bounded_possible_positions] = position_property
            sums = possibility_scores.sum(dim=1)
            valid_scores = (sums > 1e-8).nonzero(as_tuple=True)[0]
            zero_scores = (sums == 1e-8).nonzero(as_tuple=True)[0]
            selection = torch.zeros(agentIDs.shape[0], dtype=torch.long)
            if valid_scores.numel() > 0:
                selection[valid_scores] = torch.multinomial(possibility_scores[valid_scores], 1).squeeze(1)
            if zero_scores.numel() > 0:
                random_score = torch.rand(bounded_possible_positions[zero_scores].shape, dtype=torch.float16).masked_fill(~bounded_possible_positions[zero_scores], -1.0)
                selection[zero_scores] = random_score.argmax(dim=1)
                
    else:
        random_score = torch.rand(bounded_possible_positions.shape, dtype=torch.float16).masked_fill(~bounded_possible_positions, -1.0)
        selection = random_score.argmax(dim=1)
    new_positions = possible_positions[torch.arange(agentIDs.shape[0]), selection]

    return new_positions,agentIDs

def _set_agent_positions(model_graph, grid_environment, agentIDs, new_positions):
    """Set positions for specified agents in the grid environment.
    
    Args:
        model_graph (DGLGraph): All agent node and edge data
        grid_environment (GridEnvironment): agent environment information
        agentIDs (torch.Tensor): Specific agents to move
        new_positions (torch.Tensor): New positions for the specified agents
    """
    if agentIDs.shape[0] != new_positions.shape[0]:
            raise ValueError("There must be a position for each agent, but agentIDs and new_positions have different lengths.")
    if new_positions.shape[1] != 2:
            raise ValueError("Each position must have an x and y coordinate.")
    x_max = grid_environment.grid_shape[0] - 1
    y_max = grid_environment.grid_shape[1] - 1

    model_graph.ndata['x'][agentIDs] = new_positions[:, 0]
    model_graph.ndata['y'][agentIDs] = new_positions[:, 1]

    if (model_graph.ndata['x'] > x_max).any() or (model_graph.ndata['y'] > y_max).any():
        raise ValueError("One or more agent positions are out of bounds.")

def _random_jump(grid_environment, agentIDs):
    x = torch.randint(grid_environment.grid_shape[0], (agentIDs.shape[0],))
    y = torch.randint(grid_environment.grid_shape[1], (agentIDs.shape[0],))
    new_positions = torch.stack((x, y), dim=1)
    return new_positions

def _square(range_min=0, range_max=1):
    possibility_mask = torch.ones(2 * range_max + 1, 2*range_max + 1)
    possibility_mask[range_max - range_min + 1:range_max + range_min, 
                     range_max - range_min + 1:range_max + range_min] = 0
    return possibility_mask

def _circle(range_min=0, range_max=1):
    y, x = torch.meshgrid(torch.arange(range_max*2+1), 
                          torch.arange(range_max*2+1))
    possibility_mask = torch.sqrt((x - range_max).float()**2 + 
                                  (y - range_max).float()**2).round()
    possibility_mask[possibility_mask>range_max] = 0
    possibility_mask[possibility_mask<range_min] = 0
    possibility_mask[possibility_mask>0] = 1
    return possibility_mask

def _radial(range_min=0, range_max=1):
    possibility_mask = torch.zeros(2*range_max+1, 2*range_max+1)
    possibility_mask[range_max, :] = 1
    possibility_mask[:, range_max] = 1
    for i in range(2*range_max+1):
        possibility_mask[i, possibility_mask.size(1) - 1 - i] = 1
        possibility_mask[i, i] = 1
    possibility_mask[range_max-range_min+1:range_max+range_min, 
                     range_max-range_min+1:range_max+range_min] = 0
    return possibility_mask

def _cross(range_min=0, range_max=1):
    possibility_mask = torch.zeros(2*range_max+1, 2*range_max+1)
    possibility_mask[range_max, :] = 1
    possibility_mask[:, range_max] = 1
    possibility_mask[range_max-range_min+1:range_max+range_min, 
                     range_max-range_min+1:range_max+range_min] = 0
    return possibility_mask

