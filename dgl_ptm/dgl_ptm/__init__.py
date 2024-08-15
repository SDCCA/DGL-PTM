"""Documentation about dgl_ptm"""
import logging

#from dgl_ptm.model import initialize_model, step
from dgl_ptm.model.initialize_model import PovertyTrapModel
# from dgl_ptm.agent import agent_update
# from dgl_ptm.agentInteraction import trade_money
# from dgl_ptm.network import global_attachment, link_deletion
# from dgl_ptm.model import initialize_model, data_collection, step
# from dgl_ptm.util import *
from dgl_ptm import config

logging.getLogger(__name__).addHandler(logging.NullHandler())

__author__ = "Team Atlas"
__email__ = "m.grootes@esciencecenter.nl"
__version__ = "0.1.0"
