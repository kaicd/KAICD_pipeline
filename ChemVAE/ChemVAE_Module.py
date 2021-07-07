import json

import torch as th
import pytorch_lightning as pl
from pytoda.smiles.smiles_language import SMILESLanguage

from Utility.hyperparams import OPTIMIZER_FACTORY
from Utility.loss_functions import vae_loss_function
from Utility.utils import (
    packed_sequential_data_preparation,
    sequential_data_preparation,
)
from StackGRUEncoder import StackGRUEncoder
from StackGRUDecoder import StackGRUDecoder


class ChemVAE(pl.LightningModule):
    def __init__(
        self, project_filepath, params_filepath, smiles_language_filepath, **kwargs
    ):
        super(ChemVAE, self).__init__()
        # Set configuration file path
        self.params_filepath = params_filepath
        self.smiles_language_filepath = project_filepath + smiles_language_filepath
        # Load parameters
        params = {}
        with open(self.params_filepath) as f:
            params.update(json.load(f))
        # Initialize encoder and decoder
        self.encoder = StackGRUEncoder(params)
        self.decoder = StackGRUDecoder(params)
        # Set training parameters
        self.epochs = params.get("epochs", 100)
        self.opt_fn = OPTIMIZER_FACTORY[params.get("optimizer", "Adadelta")]
        self.lr = params.get("learning_rate", 0.0005)
        # Set model parameters
        self.smiles_language = SMILESLanguage.load(self.smiles_language_filepath)
        self.batch_mode = params.get("batch_mode")
        # Set forwarding parameters
        self.input_keep = params["input_keep"]
        self.start_index = 2
        self.end_index = 3
        # Set loss function parameters
        self.kl_growth = params["kl_growth"]

    def forward(self, input_seq, decoder_seq, target_seq):
        # Encoder train step
        hidden = self.encoder.init_hidden.to(self.device)
        stack = self.encoder.init_stack.to(self.device)
        hidden = getattr(self.encoder, "_forward_pass_" + self.batch_mode)(
            input_seq, hidden, stack
        )
        mu = self.encoder.hidden_to_mu(hidden)
        logvar = self.encoder.hidden_to_logvar(hidden)

        # Reparameterize step
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        latent_z = eps.mul(std).add_(mu).unsqueeze(0)

        # Decoder train step
        latent_z = latent_z.repeat(self.decoder.n_layers, 1, 1)
        hidden = self.decoder.latent_to_hidden(latent_z)
        stack = self.decoder.init_stack
        decoder_loss = getattr(self.decoder, "_forward_pass_" + self.batch_mode)(
            decoder_seq, target_seq, hidden, stack
        )

        return decoder_loss, mu, logvar

    def shared_step(self, batch, *args, **kwargs):
        _batch = (
            batch
            if self.batch_mode == "packed"
            else th.nn.utils.rnn.pad_sequence(batch).to(self.device)
        )
        _preparation = (
            packed_sequential_data_preparation
            if self.batch_mode == "packed"
            else sequential_data_preparation
        )
        encoder_seq, decoder_seq, target_seq = _preparation(
            _batch,
            self.device,
            input_keep=self.input_keep,
            start_index=self.start_index,
            end_index=self.end_index,
        )
        decoder_loss, mu, logvar = self(encoder_seq, decoder_seq, target_seq)

        return decoder_loss, mu, logvar, target_seq

    def training_step(self, batch, batch_idx, *args, **kwargs):
        decoder_loss, mu, logvar, _ = self.shared_step(batch)
        loss, kl_div = vae_loss_function(
            decoder_loss, mu, logvar, kl_growth=self.kl_growth, step=self.global_step
        )
        self.log("train_loss", loss)
        self.log("train_kl_div", kl_div)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        decoder_loss, mu, logvar, target_seq = self.shared_step(batch)

        return {
            "decoder_loss": decoder_loss,
            "mu": mu,
            "logvar": logvar,
            "target_seq": target_seq,
        }

    def validation_epoch_end(self, outputs):
        losses = []
        kl_divs = []

        for o in outputs:
            loss, kl_div = vae_loss_function(
                o["decoder_loss"], o["mu"], o["logvar"], eval_mode=True
            )
            losses.append(loss)
            kl_divs.append(kl_div)

        loss = th.mean(th.Tensor(losses).to(self.device))
        kl_div = th.mean(th.Tensor(kl_divs).to(self.device))
        self.log("val_loss", loss)
        self.log("val_kl_div", kl_div)

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
