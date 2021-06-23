import json
import pickle
from itertools import takewhile

import numpy as np
import torch as th
import pytorch_lightning as pl
import wandb
from wandb import Image
from paccmann_chemistry.utils import (
    packed_sequential_data_preparation,
    sequential_data_preparation,
    unpack_sequence,
    print_example_reconstruction,
    crop_start_stop,
)
from paccmann_chemistry.utils.hyperparams import OPTIMIZER_FACTORY, SEARCH_FACTORY
from paccmann_chemistry.utils.loss_functions import vae_loss_function
from paccmann_chemistry.utils.search import *
from pytoda.smiles.smiles_language import SMILESLanguage
from rdkit import Chem
from rdkit.Chem import Draw

from modules.StackGRUEncoder import StackGRUEncoder
from modules.StackGRUDecoder import StackGRUDecoder


class VAE(pl.LightningModule):
    def __init__(
        self, project_filepath, params_filepath, smiles_language_filepath, **kwargs
    ):
        """
        Initialization.
        Args:
            encoder (StackGRUEncoder): the encoder object.
            decoder (StackGRUDecoder): the decoder object.
        """
        super(VAE, self).__init__()
        self.params_filepath = project_filepath + params_filepath
        self.smiles_language_filepath = project_filepath + smiles_language_filepath

        # Model Parameter
        params = {}
        with open(self.params_filepath) as f:
            params.update(json.load(f))

        self.encoder = StackGRUEncoder(params)
        self.decoder = StackGRUDecoder(params)
        self.smiles_language = SMILESLanguage.load(self.smiles_language_filepath)
        self.batch_mode = params.get("batch_mode")
        self.selfies = params.get("selfies", False)
        self.data_preparation = self.get_data_preparation(self.batch_mode)
        self.search = SEARCH_FACTORY[params.get("decoder_search", "sampling")](
            temperature=params.get("temperature", 1.0),
            beam_width=params.get("beam_width", 3),
            top_tokens=params.get("top_tokens", 5),
        )  # yapf: disable
        self.opt_fn = OPTIMIZER_FACTORY[params.get("optimizer", "adadelta")]
        self.lr = params["learning_rate"]
        self.kl_growth = params["kl_growth"]
        self.input_keep = params["input_keep"]
        self.start_index = 2
        self.end_index = 3
        self.generate_len = params["generate_len"]
        self.eval_interval = params["eval_interval"]
        self.epochs = params.get("epochs", 100)
        self.lr = params.get("learning_rate", 0.0005)
        self.automatic_optimization = False

    def encode(self, input_seq):
        """
        VAE Encoder.
        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
                of shape `[max batch sequence length +1, batch_size]`, where +1
                is for the added start_index.
        Returns:
            mu (torch.Tensor): the latent mean of shape
                `[1, batch_size, latent_dim]`.
            logvar (torch.Tensor): log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        mu, logvar = self.encoder.encoder_train_step(input_seq)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        """
        Sample Z From Latent Dist.
        Args:
            mu (torch.Tensor): the latent mean of shape
                `[1, batch_size, latent_dim]`.
            logvar (torch.Tensor): log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        Returns:
            torch.Tensor: Sampled latent z from the latent distribution of
                shape `[1, batch_size, latent_dim]`.
        """
        std = th.exp(0.5 * logvar)
        eps = th.randn_like(std)
        return eps.mul(std).add_(mu)  # if self.training else mu

    def decode(self, latent_z, input_seq, target_seq):
        """
        Decode The Latent Z (for training).
        Args:
            latent_z (torch.Tensor): the sampled latent representation
                of the SMILES to be used for generation of shape
                `[1, batch_size, latent_dim]`
            input_seq (torch.Tensor): the sequence of indices for the input
                of shape `[max batch sequence length +1, batch_size]`, where +1
                is for the added start_index.
            target_seq (torch.Tensor): the sequence of indices for the
                target of shape `[max batch sequence length +1, batch_size]`,
                where +1 is for the added end_index.
        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.
        Returns:
            the cross-entropy training loss for the decoder.
        """
        n_layers = self.decoder.n_layers
        latent_z = latent_z.repeat(n_layers, 1, 1)
        decoder_loss = self.decoder.decoder_train_step(latent_z, input_seq, target_seq)
        return decoder_loss

    def generate(
        self,
        latent_z,
        prime_input,
        end_token,
        generate_len=100,
        search=SamplingSearch(),
    ):
        """
        Generate SMILES From Latent Z.
        Args:
            latent_z (torch.Tensor): The sampled latent representation
                of size `[1, batch_size, latent_dim]`.
            prime_input (torch.Tensor): Tensor of indices for the priming
                string. Must be of size `[1, prime_input length]` or
                `[prime_input length]`.
                Example:
                    `prime_input = torch.tensor([2, 4, 5]).view(1, -1)`
                    or
                    `prime_input = torch.tensor([2, 4, 5])`
            end_token (torch.Tensor): End token for the generated molecule
                of shape `[1]`.
                Example: `end_token = torch.LongTensor([3])`
            generate_len (int): Length of the generated molecule.
        Returns:
            iterable: An iterator returning the torch tensor of
                sequence(s) for the generated molecule(s) of shape
                `[sequence length]`.
        Note: The start and end tokens are automatically stripped
            from the returned torch tensors for the generated molecule.
        """
        generated_batch = self.decoder.generate_from_latent(
            latent_z, prime_input, end_token, search=search, generate_len=generate_len
        )

        molecule_gen = (
            takewhile(lambda x: x != end_token.cpu(), molecule[1:].cpu())
            for molecule in generated_batch
        )  # yapf: disable

        molecule_map = map(list, molecule_gen)
        molecule_iter = iter(map(th.tensor, molecule_map))

        return molecule_iter

    def _prepare_packed(self, batch, input_keep, start_index, end_index, device):
        encoder_seq, decoder_seq, target_seq = packed_sequential_data_preparation(
            batch, input_keep=input_keep, start_index=start_index, end_index=end_index
        )

        return encoder_seq, decoder_seq, target_seq

    def _prepare_padded(self, batch, input_keep, start_index, end_index, device):
        padded_batch = th.nn.utils.rnn.pad_sequence(batch)
        padded_batch = padded_batch.to(device)
        encoder_seq, decoder_seq, target_seq = sequential_data_preparation(
            padded_batch,
            input_keep=input_keep,
            start_index=start_index,
            end_index=end_index,
        )
        return encoder_seq, decoder_seq, target_seq

    def get_data_preparation(self, mode):
        """Select data preparation function mode
        Args:
            mode (str): Mode to use. Available modes:
                `Padded`, `Packed`
        """
        if not isinstance(mode, str):
            raise TypeError("Argument `mode` should be a string.")
        mode = mode.capitalize()
        MODES = {"Padded": self._prepare_padded, "Packed": self._prepare_packed}
        if mode not in MODES:
            raise NotImplementedError(
                f"Unknown mode: {mode}. Available modes: {MODES.keys()}"
            )
        return MODES[mode]

    def forward(self, input_seq, decoder_seq, target_seq):
        """
        The Forward Function.
        Args:
            input_seq (torch.Tensor): the sequence of indices for the input
                of shape `[max batch sequence length +1, batch_size]`, where +1
                is for the added start_index.
            target_seq (torch.Tensor): the sequence of indices for the
                target of shape `[max batch sequence length +1, batch_size]`,
                where +1 is for the added end_index.
        Note: Input and target sequences are outputs of
            sequential_data_preparation(batch) with batches returned by a
            DataLoader object.
        Returns:
            (torch.Tensor, torch.Tensor, torch.Tensor): decoder_loss, mu,
                logvar
            decoder_loss is the cross-entropy training loss for the decoder.
            mu is the latent mean of shape `[1, batch_size, latent_dim]`.
            logvar is log of the latent variance of shape
                `[1, batch_size, latent_dim]`.
        """
        mu, logvar = self.encode(input_seq)
        latent_z = self.reparameterize(mu, logvar).unsqueeze(0)
        decoder_loss = self.decode(latent_z, decoder_seq, target_seq)
        return decoder_loss, mu, logvar

    def training_step(self, batch, batch_idx, *args, **kwargs):
        opt = self.optimizers()
        encoder_seq, decoder_seq, target_seq = self.data_preparation(
            batch,
            input_keep=self.input_keep,
            start_index=self.start_index,
            end_index=self.end_index,
            device=self.device,
        )

        self.train()
        opt.zero_grad()
        decoder_loss, mu, logvar = self(encoder_seq, decoder_seq, target_seq)
        loss, kl_div = vae_loss_function(
            decoder_loss, mu, logvar, kl_growth=self.kl_growth, step=self.global_step
        )
        self.log("train_loss", loss)
        self.log("train_kl_div", kl_div)
        self.manual_backward(loss)
        opt.step()

        if batch_idx % self.eval_interval == 0:
            self.eval()
            latent_z = th.randn(1, mu.shape[0], mu.shape[1]).to(self.device)
            molecule_iter = self.generate(
                latent_z,
                prime_input=th.tensor([self.start_index]).to(self.device),
                end_token=th.tensor([self.end_index]).to(self.device),
                generate_len=self.generate_len,
                search=self.search,
            )
            mol = next(molecule_iter)
            mol = self.smiles_language.token_indexes_to_smiles(
                crop_start_stop(mol, self.smiles_language)
            )
            mol = self.smiles_language.selfies_to_smiles(mol) if self.selfies else mol

            if not mol == -1:
                mol = Chem.MolFromSmiles(mol)
                if mol:
                    mol = np.array(Draw.MolsToImage([mol]))
                    self.logger.experiment.log(
                        {
                            "generated_mol_img": [
                                Image(mol, caption="generated_mol_img")
                            ]
                        }
                    )

            if self.batch_mode == "packed":
                target_seq = unpack_sequence(target_seq)

            target, pred = print_example_reconstruction(
                self.decoder.outputs, target_seq, self.smiles_language, self.selfies
            )

            mol = None
            molt = None
            if not target == -1:
                mol = Chem.MolFromSmiles(target)
            if not pred == -1:
                molt = Chem.MolFromSmiles(pred)

            if mol and molt:
                mols = np.array(Draw.MolsToImage([mol, molt]))
                self.logger.experiment.log(
                    {
                        "reconstructed_mol_img": [
                            Image(mols, caption="reconstructed_mol_img")
                        ]
                    }
                )
    def validation_step(self, batch, *args, **kwargs):
        self.eval()
        encoder_seq, decoder_seq, target_seq = self.data_preparation(
            batch,
            input_keep=self.input_keep,
            start_index=self.start_index,
            end_index=self.end_index,
            device=self.device,
        )

        decoder_loss, mu, logvar = self(encoder_seq, decoder_seq, target_seq)

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

    def _associate_language(self, language):
        """
        Bind a SMILES language object to the model.
        Arguments:
            language  [pytoda.smiles.smiles_language.SMILESLanguage] --
                A SMILES language object either supporting SMILES or SELFIES
        Raises:
            TypeError:
        """
        if isinstance(language, pytoda.smiles.smiles_language.SMILESLanguage):
            self.smiles_language = language

        else:
            raise TypeError(
                "Please insert a smiles language (object of type "
                "pytoda.smiles.smiles_language.SMILESLanguage . Given was "
                f"{type(language)}"
            )
