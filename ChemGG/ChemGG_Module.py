import json

import torch as th
import pytorch_lightning as pl

from ChemGG import MPNN
from ChemGG.ChemGG_Analyzer import ChemGG_Analyzer
from Utility.hyperparams import OPTIMIZER_FACTORY
from Utility.loss_functions import gg_loss
from Utility.utils import update_features

class ChemGG_Module(pl.LightningModule):
    def __init__(
        self,
        model_name: str,
        params_filepath: str,
        analyzer: ChemGG_Analyzer
    ):
        super(ChemGG_Module, self).__init__()
        # load model parameters
        self.params = {}
        with open(params_filepath) as f:
            params = json.load(f)
            self.params.update(params["CONFIG"])
            self.params.update(params[model_name])
        # update feature parameters
        self.params.update(update_features(self.params))
        # set optimizer
        self.opt_fn = OPTIMIZER_FACTORY[self.params.get("optimizer", "Adam")]
        # load model
        self.model = getattr(MPNN, model_name)(params=self.params)

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

    def configure_optimizers(self):
        opt = self.opt_fn(self.parameters(), lr=self.params.get("init_lr", 1e-4))
        scheduler = {
            "scheduler": th.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lambda epoch: max(1e-7, 1 - epoch / self.epochs),
            ),
            "reduce_on_plateau": False,
            "interval": "epoch",
            "frequency": 1,
        }

        return [opt], [scheduler]