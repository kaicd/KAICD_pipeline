import json

import torch as th
import pytorch_lightning as pl

from gnn.mpnn import *


class ChemGG_Module(pl.LightningModule):
    def __init__(self, model_name, params_filepath):
        super(ChemGG_Module, self).__init__()
        # load parameters
        params = {}
        with open(params_filepath) as f:
            params.update(json.load(f))

        # load model
