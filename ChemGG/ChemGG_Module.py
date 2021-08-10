import json

import torch as th
import pytorch_lightning as pl

from .ChemGG_Analyzer import ChemGG_Analyzer
from .ChemGG_Generator import ChemGG_Generator
from ChemGG import ChemGG_MPNN
from Utility.hyperparams import OPTIMIZER_FACTORY
from Utility.utils import update_features
from Utility.loss_functions import gg_loss


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
        self.model = getattr(ChemGG_MPNN, model_name)(params=self.params)
        self.analyzer = analyzer

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

        # Evaluating model
        self.generate_graphs(n_samples=self.params["n_samples"], evaluation=True)
        print("* Evaluating model.", flush=True)
        self.analyzer.model = self.model
        self.analyzer.evaluate_model(nll_per_action=self.nll_per_action)

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

    def generate_graphs(self, n_samples: int, evaluation: bool = False) -> None:
        """
        Generates molecular graphs and evaluates them. Generates the graphs in batches of either
        the size of the mini-batches or `n_samples`, whichever is smaller.
        Args:
        ----
            n_samples (int) : How many graphs to generate.
            evaluation (bool) : Indicates whether the model will be evaluated, in which
              case we will also need the NLL per action for the generated graphs.
        """
        print(f"* Generating {n_samples} molecules.", flush=True)
        generation_batch_size = min(self.params["batch_size"], n_samples)
        n_generation_batches = int(n_samples / generation_batch_size)

        generator = ChemGG_Generator(model=self.model, batch_size=generation_batch_size)

        # generate graphs in batches
        for idx in range(0, n_generation_batches + 1):
            print("Batch", idx, "of", n_generation_batches)

            # generate one batch of graphs
            graphs, action_nlls, final_nlls, termination = generator.sample()

            # analyze properties of new graphs and save results
            self.analyzer.evaluate_generated_graphs(generated_graphs=graphs,
                                                    termination=termination,
                                                    nlls=final_nlls,
                                                    ts_properties=self.ts_properties,
                                                    generation_batch_idx=idx)

            # keep track of NLLs per action; note that only NLLs for the first batch are kept,
            # as only a few are needed to evaluate the model (more efficient than saving all)
            if evaluation and idx == 0:
                self.nll_per_action = action_nlls