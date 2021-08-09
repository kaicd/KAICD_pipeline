import json

import torch as th
import pytorch_lightning as pl

from .ChemGG_Base import ChemGG_Base
from .ChemGG_Analyzer import ChemGG_Analyzer
from ChemGG import ChemGG_MPNN
from Utility.hyperparams import OPTIMIZER_FACTORY
from Utility.loss_functions import gg_loss
from Utility.utils import update_features


class ChemGG_Module(ChemmGG_Base):
    def __init__(self, **kwargs):
        super(ChemGG_Module, self).__init__(**kwargs)

    def forward(self, nodes, edges):
        return self.model(nodes, edges)

    def training_step(self, batch, *args, **kwargs):
        nodes, edges, target_output = batch
        output = self(nodes, edges)

        loss = gg_loss(output, target_output)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        nodes, edges, target_output = batch
        output = self(nodes, edges)

        return {"output": output, "target_output": target_output}

    def validation_epoch_end(self, outputs):
        losses = []
        for o in outputs:
            losses.append(gg_loss(o["output"], o["target_output"]))

        loss = th.mean(th.Tensor(losses).to(self.device))
        self.log("val_loss", loss)

    def on_train_end(self):
        pass


