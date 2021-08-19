import json
from collections import OrderedDict

import pytoda
import pytorch_lightning as pl
import torch as th
import torch.nn as nn
from pytoda.smiles.transforms import AugmentTensor

from Utility.hyperparams import (
    ACTIVATION_FN_FACTORY,
    LOSS_FN_FACTORY,
    OPTIMIZER_FACTORY,
)
from Utility.interpret import (
    monte_carlo_dropout,
    test_time_augmentation,
)
from Utility.layers import (
    convolutional_layer,
    dense_attention_layer,
    dense_layer,
    ContextAttentionLayer,
)
from Utility.loss_functions import pearsonr
from Utility.utils import get_log_molar


class PredictorDS_Module(pl.LightningModule):
    """Multiscale Convolutional Attentive Encoder.
    This is the MCA model as presented in the authors publication in
    Molecular Pharmaceutics:
        https://pubs.acs.org/doi/10.1021/acs.molpharmaceut.9b00520.
    """

    def __init__(self, params_filepath, *args, **kwargs):
        super(PredictorDS_Module, self).__init__()

        # Model Parameter
        params = {}
        with open(params_filepath) as f:
            params.update(json.load(f))

        self.params = params
        self.loss_fn = LOSS_FN_FACTORY[params.get("loss_fn", "mse")]
        self.opt_fn = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")]
        self.act_fn = ACTIVATION_FN_FACTORY[params.get("activation_fn", "relu")]
        self.lr = params.get("lr", 0.01)
        self.epochs = params["epochs"]
        self.min_max_scaling = (
            True
            if params.get("drug_sensitivity_processing_parameters", {}) != {}
            else False
        )
        if self.min_max_scaling:
            self.IC50_max = params["drug_sensitivity_processing_parameters"][
                "parameters"
            ]["max"]
            self.IC50_min = params["drug_sensitivity_processing_parameters"][
                "parameters"
            ]["min"]
        # Model inputs
        self.number_of_genes = params.get("number_of_genes", 2128)
        self.smiles_attention_size = params.get("smiles_attention_size", 64)
        # Model architecture (hyperparameter)
        self.multiheads = params.get("multiheads", [4, 4, 4, 4])
        self.filters = params.get("filters", [64, 64, 64])
        self.hidden_sizes = [
            self.multiheads[0] * params["smiles_embedding_size"]
            + sum([h * f for h, f in zip(self.multiheads[1:], self.filters)])
        ] + params.get("stacked_dense_hidden_sizes", [1024, 512])

        if params.get("gene_to_dense", False):  # Optional skip connection
            self.hidden_sizes[0] += self.number_of_genes
        self.dropout = params.get("dropout", 0.5)
        self.temperature = params.get("temperature", 1.0)
        self.kernel_sizes = params.get(
            "kernel_sizes",
            [
                [3, params["smiles_embedding_size"]],
                [5, params["smiles_embedding_size"]],
                [11, params["smiles_embedding_size"]],
            ],
        )
        if len(self.filters) != len(self.kernel_sizes):
            raise ValueError("Length of filter and kernel size lists do not match.")
        if len(self.filters) + 1 != len(self.multiheads):
            raise ValueError("Length of filter and multihead lists do not match")

        # Build the model
        self.smiles_embedding = nn.Embedding(
            self.params["smiles_vocabulary_size"],
            self.params["smiles_embedding_size"],
            scale_grad_by_freq=params.get("embed_scale_grad", False),
        )
        self.gene_attention_layer = dense_attention_layer(
            self.number_of_genes, temperature=self.temperature, dropout=self.dropout
        ).to(self.device)

        self.convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"convolutional_{index}",
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            batch_norm=params.get("batch_norm", False),
                            dropout=self.dropout,
                        ).to(self.device),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.filters, self.kernel_sizes)
                    )
                ]
            )
        )

        smiles_hidden_sizes = [params["smiles_embedding_size"]] + self.filters

        self.context_attention_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"context_attention_{layer}_head_{head}",
                        ContextAttentionLayer(
                            smiles_hidden_sizes[layer],
                            42,  # Can be anything since context is only 1D (omic)
                            self.number_of_genes,
                            attention_size=self.smiles_attention_size,
                            individual_nonlinearity=params.get(
                                "context_nonlinearity", nn.Sequential()
                            ),
                        ),
                    )
                    for layer in range(len(self.multiheads))
                    for head in range(self.multiheads[layer])
                ]
            )
        )  # yapf: disable

        # Only applied if params['batch_norm'] = True
        self.batch_norm = nn.BatchNorm1d(self.hidden_sizes[0])
        self.dense_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        "dense_{}".format(ind),
                        dense_layer(
                            self.hidden_sizes[ind],
                            self.hidden_sizes[ind + 1],
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=params.get("batch_norm", True),
                        ).to(self.device),
                    )
                    for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        self.final_dense = (
            nn.Linear(self.hidden_sizes[-1], 1)
            if not params.get("final_activation", False)
            else nn.Sequential(
                OrderedDict(
                    [
                        ("projection", nn.Linear(self.hidden_sizes[-1], 1)),
                        ("sigmoidal", ACTIVATION_FN_FACTORY["sigmoid"]),
                    ]
                )
            )
        )

    def forward(self, smiles, gep, confidence=False):
        """Forward pass through the MCA.
        Args:
            smiles (torch.Tensor): of type int and shape `[bs, seq_length]`.
            gep (torch.Tensor): of shape `[bs, num_genes]`.
            confidence (bool, optional) whether the confidence estimates are
                performed.
        Returns:
            (torch.Tensor, torch.Tensor): predictions, prediction_dict
            predictions is IC50 drug sensitivity prediction of shape `[bs, 1]`.
            prediction_dict includes the prediction and attention weights.
        """
        embedded_smiles = self.smiles_embedding(smiles.to(dtype=th.int64))

        # Gene attention weights
        gene_alphas = self.gene_attention_layer(gep)

        # Filter the gene expression with the weights.
        encoded_genes = gene_alphas * gep

        # NOTE: SMILES Convolutions. Unsqueeze has shape bs x 1 x T x H.
        encoded_smiles = [embedded_smiles] + [
            self.convolutional_layers[ind](th.unsqueeze(embedded_smiles, 1)).permute(
                0, 2, 1
            )
            for ind in range(len(self.convolutional_layers))
        ]

        # NOTE: SMILES Attention mechanism
        encodings, smiles_alphas = [], []
        context = th.unsqueeze(encoded_genes, 1)
        for layer in range(len(self.multiheads)):
            for head in range(self.multiheads[layer]):
                ind = self.multiheads[0] * layer + head
                e, a = self.context_attention_layers[ind](
                    encoded_smiles[layer], context
                )
                encodings.append(e)
                smiles_alphas.append(a)

        encodings = th.cat(encodings, dim=1)
        if self.params.get("gene_to_dense", False):
            encodings = th.cat([encodings, gep], dim=1)

        # Apply batch normalization if specified
        inputs = (
            self.batch_norm(encodings)
            if self.params.get("batch_norm", False)
            else encodings
        )
        # NOTE: stacking dense layers as a bottleneck
        for dl in self.dense_layers:
            inputs = dl(inputs)

        predictions = self.final_dense(inputs)

        prediction_dict = {}

        if not self.training:
            # The below is to ease postprocessing
            smiles_attention_weights = th.mean(
                th.cat([th.unsqueeze(p, -1) for p in smiles_alphas], dim=-1), dim=-1
            )
            prediction_dict.update(
                {
                    "gene_attention": gene_alphas,
                    "smiles_attention": smiles_attention_weights,
                    "IC50": predictions,
                    "log_micromolar_IC50": get_log_molar(
                        predictions, ic50_max=self.IC50_max, ic50_min=self.IC50_min
                    )
                    if self.min_max_scaling
                    else predictions,
                }
            )

            if confidence:
                augmenter = AugmentTensor(self.smiles_language)
                epistemic_conf = monte_carlo_dropout(
                    self, regime="tensors", tensors=(smiles, gep), repetitions=5
                )
                aleatoric_conf = test_time_augmentation(
                    self,
                    regime="tensors",
                    tensors=(smiles, gep),
                    repetitions=5,
                    augmenter=augmenter,
                    tensors_to_augment=0,
                )

                prediction_dict.update(
                    {
                        "epistemic_confidence": epistemic_conf,
                        "aleatoric_confidence": aleatoric_conf,
                    }
                )

        return predictions, prediction_dict

    def training_step(self, batch, *args, **kwargs):
        smiles, gep, y = batch
        y_hat, _ = self(th.squeeze(smiles), gep)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        smiles, gep, y = batch
        y_hat, _ = self(th.squeeze(smiles), gep)

        return {"y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs):
        y_hats = []
        ys = []

        for o in outputs:
            y_hats.append(o["y_hat"])
            ys.append(o["y"])

        y_hat = th.cat(y_hats)
        y = th.cat(ys)
        loss = self.loss_fn(y_hat, y)
        pearson = pearsonr(th.squeeze(y_hat), th.squeeze(y))
        rmse = th.sqrt(th.mean((y_hat - y) ** 2))

        self.log("val_loss", loss)
        self.log("val_pearson", pearson)
        self.log("val_rmse", rmse)

    def configure_optimizers(self):
        opt = self.opt_fn(self.parameters(), lr=self.lr)
        sched = {
            "scheduler": th.optim.lr_scheduler.LambdaLR(
                opt,
                lr_lambda=lambda epoch: max(1e-7, 1 - epoch / self.epochs),
            ),
            "reduce_on_plateau": False,
            "interval": "epoch",
            "frequency": 1,
        }

        return [opt], [sched]

    def _associate_language(self, smiles_language):
        """
        Bind a SMILES language object to the model. Is only used inside the
        confidence estimation.
        Arguments:
            smiles_language {[pytoda.smiles.smiles_language.SMILESLanguage]}
            -- [A SMILES language object]
        Raises:
            TypeError:
        """
        if not isinstance(
            smiles_language, pytoda.smiles.smiles_language.SMILESLanguage
        ):
            raise TypeError(
                "Please insert a smiles language (object of type "
                "pytoda.smiles.smiles_language.SMILESLanguage). Given was "
                f"{type(smiles_language)}"
            )
        self.smiles_language = smiles_language
