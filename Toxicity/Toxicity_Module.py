import json
import pickle
from collections import OrderedDict
from typing import Tuple

import pytorch_lightning as pl
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import (
    auc,
    average_precision_score,
    roc_curve,
)

from Utility.hyperparams import (
    ACTIVATION_FN_FACTORY,
    OPTIMIZER_FACTORY,
    LOSS_FN_FACTORY,
)
from Utility.layers import (
    alpha_projection,
    convolutional_layer,
    dense_layer,
    smiles_projection,
    EnsembleLayer,
)


class Toxicity_Module(pl.LightningModule):
    """
    Multiscale Convolutional Attentive Encoder.
    This is the MCA model similiar to the one presented in publication in
    Molecular Pharmaceutics https://arxiv.org/abs/1904.11223.
    Differences:
        - uses self instead of context attention since input is unimodal.
        - MultiLabel classification implementation (sigmoidal in last layer)
    """

    def __init__(
        self,
        params_filepath,
        **kwargs,
    ):
        super(Toxicity_Module, self).__init__()
        self.params_filepath = params_filepath

        # Model Parameter
        params = {}
        with open(self.params_filepath) as f:
            params.update(json.load(f))

        self.lr = params.get("lr", 0.00001)
        self.epochs = params["epochs"]
        self.loss_fn = LOSS_FN_FACTORY[
            params.get("loss_fn", "binary_cross_entropy_ignore_nan_and_sum")
        ]  # yapf: disable
        self.opt_fn = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")]

        self.num_tasks = params.get("num_tasks", 12)
        self.smiles_attention_size = params.get("smiles_attention_size", 64)

        # Model architecture (hyperparameter)
        self.multiheads = params.get("multiheads", [4, 4, 4, 4])
        self.filters = params.get("filters", [64, 64, 64])
        self.hidden_sizes = [
            self.multiheads[0] * params["smiles_embedding_size"]
            + sum([h * f for h, f in zip(self.multiheads[1:], self.filters)])
        ] + params.get("stacked_dense_hidden_sizes", [1024, 512])

        self.dropout = params.get("dropout", 0.5)
        self.use_batch_norm = params.get("batch_norm", True)
        self.act_fn = ACTIVATION_FN_FACTORY[params.get("activation_fn", "relu")]
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

        # Build the model. First the embeddings
        if params.get("embedding", "learned") == "learned":

            self.smiles_embedding = nn.Embedding(
                params["smiles_vocabulary_size"],
                params["smiles_embedding_size"],
                scale_grad_by_freq=params.get("embed_scale_grad", False),
            )
        elif params.get("embedding", "learned") == "one_hot":
            self.smiles_embedding = nn.Embedding(
                params["smiles_vocabulary_size"], params["smiles_vocabulary_size"]
            )
            # Plug in one hot-vectors and freeze weights
            self.smiles_embedding.load_state_dict(
                {"weight": F.one_hot(th.arange(params["smiles_vocabulary_size"]))}
            )
            self.smiles_embedding.weight.requires_grad = False

        elif params.get("embedding", "learned") == "pretrained":
            # Load the pretrained embeddings
            try:
                with open(params["embedding_path"], "rb") as f:
                    embeddings = pickle.load(f)
            except KeyError:
                raise KeyError("Path for embeddings is missing in params.")

            # Plug into layer
            self.smiles_embedding = nn.Embedding(
                embeddings.shape[0], embeddings.shape[1]
            )
            self.smiles_embedding.load_state_dict({"weight": th.Tensor(embeddings)})
            if params.get("fix_embeddings", True):
                self.smiles_embedding.weight.requires_grad = False

        else:
            raise ValueError(f"Unknown embedding type: {params['embedding']}")

        self.convolutional_layers = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"convolutional_{index}",
                        convolutional_layer(
                            num_kernel,
                            kernel_size,
                            act_fn=self.act_fn,
                            dropout=self.dropout,
                            batch_norm=self.use_batch_norm,
                        ),
                    )
                    for index, (num_kernel, kernel_size) in enumerate(
                        zip(self.filters, self.kernel_sizes)
                    )
                ]
            )
        )

        smiles_hidden_sizes = [params["smiles_embedding_size"]] + self.filters
        self.smiles_projections = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"smiles_projection_{self.multiheads[0]*layer+index}",
                        smiles_projection(
                            smiles_hidden_sizes[layer], self.smiles_attention_size
                        ),
                    )
                    for layer in range(len(self.multiheads))
                    for index in range(self.multiheads[layer])
                ]
            )
        )
        self.alpha_projections = nn.Sequential(
            OrderedDict(
                [
                    (
                        f"alpha_projection_{self.multiheads[0]*layer+index}",
                        alpha_projection(self.smiles_attention_size),
                    )
                    for layer in range(len(self.multiheads))
                    for index in range(self.multiheads[layer])
                ]
            )
        )

        if self.use_batch_norm:
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
                            batch_norm=self.use_batch_norm,
                        ),
                    )
                    for ind in range(len(self.hidden_sizes) - 1)
                ]
            )
        )

        if params.get("ensemble", "None") not in ["score", "prob", "None"]:
            raise NotImplementedError(
                "Choose ensemble type from ['score', 'prob', 'None']"
            )
        if params.get("ensemble", "None") == "None":
            params["ensemble_size"] = 1

        self.final_dense = EnsembleLayer(
            typ=params.get("ensemble", "score"),
            input_size=self.hidden_sizes[-1],
            output_size=self.num_tasks,
            ensemble_size=params.get("ensemble_size", 5),
            fn=ACTIVATION_FN_FACTORY["sigmoid"],
        )

        # Set class weights manually
        if "binary_cross_entropy_ignore_nan" in params.get(
            "loss_fn", "binary_cross_entropy_ignore_nan_and_sum"
        ):
            self.loss_fn.class_weights = params.get("class_weights", [1, 1])

    def forward(self, smiles: th.Tensor) -> Tuple[th.Tensor, dict]:
        """Forward pass through the MCA.
        Args:
            smiles (torch.Tensor): type int and shape: [batch_size, seq_length]
        Returns:
            (torch.Tensor, dict): predictions, prediction_dict
            predictions are toxicity predictions of shape `[bs, num_tasks]`.
            prediction_dict includes the prediction and attention weights.
        """
        embedded_smiles = self.smiles_embedding(smiles.to(dtype=th.int64))
        # SMILES Convolutions. Unsqueeze has shape batch_size x 1 x T x H.
        encoded_smiles = [embedded_smiles] + [
            self.convolutional_layers[ind](th.unsqueeze(embedded_smiles, 1)).permute(
                0, 2, 1
            )
            for ind in range(len(self.convolutional_layers))
        ]

        # NOTE: SMILES Self Attention mechanism (see )
        smiles_alphas, encodings = [], []
        for layer in range(len(self.multiheads)):
            for head in range(self.multiheads[layer]):

                ind = self.multiheads[0] * layer + head
                smiles_alphas.append(
                    self.alpha_projections[ind](
                        self.smiles_projections[ind](encoded_smiles[layer])
                    )
                )
                # Sequence is always reduced.
                encodings.append(
                    th.sum(
                        encoded_smiles[layer] * th.unsqueeze(smiles_alphas[-1], -1), 1
                    )
                )
        encodings = th.cat(encodings, dim=1)

        # Apply batch normalization if specified
        if self.use_batch_norm:
            encodings = self.batch_norm(encodings)
        for dl in self.dense_layers:
            encodings = dl(encodings)

        predictions = self.final_dense(encodings)
        prediction_dict = {
            "smiles_attention": smiles_alphas,
            "toxicities": predictions,
        }
        return predictions, prediction_dict

    def training_step(self, batch, *args, **kwargs):
        smiles, y = batch
        y_hat, _ = self(smiles)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        smiles, y = batch
        y_hat, _ = self(smiles)

        return {"y_hat": y_hat, "y": y}

    def validation_epoch_end(self, outputs):
        y_hats = []
        ys = []

        for o in outputs:
            y_hats.append(o["y_hat"])
            ys.append(o["y"])

        y_hat = th.cat(y_hats)
        y = th.cat(ys)
        y_hat = y_hat[~th.isnan(y)]
        y = y[~th.isnan(y)]
        loss = self.loss_fn(y_hat, y)
        fpr, tpr, _ = roc_curve(y.detach().cpu(), y_hat.detach().cpu())
        roc_auc = auc(fpr, tpr)
        avg_precision = average_precision_score(y.detach().cpu(), y_hat.detach().cpu())

        self.log("val_loss", loss)
        self.log("val_roc_auc", roc_auc)
        self.log("val_avg_precision_score", avg_precision)

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
