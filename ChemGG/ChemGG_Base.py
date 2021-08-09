import json

import torch as th
import pytorch_lightning as pl


class ChemGG_Base(pl.LightningModule):
    def __init__(self):
        super(ChemGG_Base, self).__init__()
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
        self.model = getattr(ChemGG_MPNN, model_name)(params=self.params)

    def training_step(self, batch, *args, **kwargs):
        return NotImplementedError

    def training_epoch_end(self, outputs):
        return NotImplementedError

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
