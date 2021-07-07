import json

import torch as th
import pytorch_lightning as pl

from DenseEncoder import DenseEncoder
from DenseDecoder import DenseDecoder
from Utility.hyperparams import LOSS_FN_FACTORY
from Utility.hyperparams import OPTIMIZER_FACTORY
from Utility.loss_functions import joint_loss
from Utility.utils import augment


class ProtVAE(pl.LightningModule):
    """Variational Auto-Encoder (VAE)"""

    def __init__(self, project_filepath, params_filepath, *args, **kwargs):
        """
        This class specifies a Variational Auto-Encoder (VAE) which
            can be instantiated with different encoder or decoder
            objects.
        Args:
            params (dict): A dict with the model parameters (<dict>).
            encoder (Encoder): An encoder object.
            decoder (Decoder): A decoder object.
         NOTE: This VAE class assumes that:
            1) The latent space should follow a multivariate unit Gaussian.
            2) The encoder strives to learn mean and log-variance of the
                latent space.
        """
        super(ProtVAE, self).__init__()
        # Set configuration file path
        self.params_filepath = params_filepath
        # Load parameters
        params = {}
        with open(self.params_filepath) as f:
            params.update(json.load(f))
        # Initialize encoder and decoder
        self.encoder = DenseEncoder(params)
        self.decoder = DenseDecoder(params)
        # Set training parameters
        self.epochs = params.get("epochs", 100)
        self.opt_fn = OPTIMIZER_FACTORY[params.get("optimizer", "Adam")]
        self.lr = params.get("lr", 0.0005)
        # Set data augmentation parameters
        self.DAE_mask = params.get("DAE_mask", 0.0)
        self.DAE_noise = params.get("DAE_noise", 0.0)
        # Set loss function parameters
        self.reconstruction_loss = LOSS_FN_FACTORY[
            params.get("reconstruction_loss", "mse")
        ]
        self.kld_loss = LOSS_FN_FACTORY[params.get("kld_loss", "kld")]
        self.alpha = params.get("alpha", 0.5)
        self.beta = params.get("beat", 1.0)
        self.kl_annealing = params.get("kl_annealing", 2)
        self.alphas = th.cat(
            [
                th.linspace(0.9999, self.alpha, self.kl_annealing),
                self.alpha * th.ones(self.epochs - self.kl_annealing),
            ]
        )
        self._assertion_tests()

    def forward(self, data):
        """The Forward Function passing data through the entire VAE.
        Args:
            data (torch.Tensor): Input data of shape
                `[batch_size, input_size]`.
        Returns:
            (torch.Tensor): A (realistic) sample decoded from the latent
                representation of length  input_size]`. Ideally data == sample.
        """
        # Encoder train step
        self.mu, self.logvar = self.encoder(data)
        # Reparameterize
        latent_z = th.randn_like(self.mu).mul_(th.exp(0.5 * self.logvar)).add_(self.mu)
        # Decoder train step
        sample = self.decoder(latent_z)

        return sample

    def shared_step(self, batch, mode, *args, **kwargs):
        _batch = (
            augment(batch, dropout=self.DAE_mask, sigma=self.DAE_noise).to(self.device)
            if mode == "train"
            else batch
        )
        _batch_fake = self(_batch).to(th.float32)
        loss, rec, kld = joint_loss(
            _batch_fake,
            batch,
            self.reconstruction_loss,
            self.kld_loss,
            self.mu,
            self.logvar,
            self.alphas[self.current_epoch] if mode == "train" else self.alpha,
            self.beta,
        )
        return loss, rec, kld

    def training_step(self, batch, *args, **kwargs):
        loss, rec, kld = self.shared_step(batch.to(th.float32), mode="train")
        self.log("train_loss", loss)
        self.log("train_rec", rec)
        self.log("train_kl_div", kld)

        return loss

    def validation_step(self, batch, *args, **kwargs):
        loss, rec, kld = self.shared_step(batch.to(th.float32), mode="valid")

        return {"loss": loss, "rec": rec, "kl_div": kld}

    def validation_epoch_end(self, outputs):
        losses = []
        recs = []
        klds = []

        for o in outputs:
            losses.append(o["loss"])
            recs.append(o["rec"])
            klds.append(o["kl_div"])

        loss = th.mean(th.Tensor(losses).to(self.device))
        rec = th.mean(th.Tensor(recs).to(self.device))
        kld = th.mean(th.Tensor(klds).to(self.device))
        self.log("val_loss", loss)
        self.log("val_rec", rec)
        self.log("val_kl_div", kld)

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

    def _assertion_tests(self):
        pass
