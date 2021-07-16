"""PaccMann^RL: Policy gradient class"""
import json
import os
import argparse

import torch as th
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from rdkit import Chem
from pytoda.transforms import LeftPadding, ToTensor
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage

from Utility.utils import ProteinDataset
from ChemVAE.ChemVAE_Module import ChemVAE_Module
from ProtVAE.ProtVAE_Module import ProtVAE_Module
from Predictor.Predictor_Module import Predictor_Module
from Utility.drug_evaluators import (
    QED,
    SCScore,
    ESOL,
    SAS,
    Lipinski,
    Tox21,
    SIDER,
    ClinTox,
    OrganDB,
)
from Utility.hyperparams import OPTIMIZER_FACTORY
from Utility.search import SamplingSearch


class Reinforce_Base(pl.LightningModule):
    """
    Pipeline to reproduce the results using pytorch_lightning of the paper
    Data-driven molecular design for discovery and synthesis of novel ligands:
    a case study on SARS-CoV-2(Machine Learning: Science and Technology, 2021).
    """

    @classmethod
    def add_model_args(cls, parent_parser: argparse.ArgumentParser):
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--mol_model_path",
            type=str,
            help="Path to chemistry model",
            default="ChemVAE/checkpoint/",
        )
        parser.add_argument(
            "--protein_model_path",
            type=str,
            help="Path to protein model",
            default="ProtVAE/checkpoint/",
        )
        parser.add_argument(
            "--affinity_model_path",
            type=str,
            help="Path to pretrained affinity model",
            default="Predictor/checkpoint/",
        )
        parser.add_argument(
            "--params_path",
            type=str,
            help="Model params json file directory",
            default="Config/Reinforce.json",
        )
        parser.add_argument(
            "--unbiased_predictions_path",
            type=str,
            help="Path to folder with aff. preds for 3000 mols from unbiased generator",
            default="/raid/PaccMann_sarscov2/data/training/unbiased_predictions",
        )
        parser.add_argument(
            "--tox21_path",
            type=str,
            help="Optional path to Tox21 model.",
        )
        parser.add_argument(
            "--organdb_path",
            type=str,
            help="Optional path to OrganDB model.",
        )
        parser.add_argument(
            "--site",
            type=str,
            help="Specify a site in case of using a OrganDB model.",
        )
        parser.add_argument(
            "--clintox_path",
            type=str,
            help="Optional path to ClinTox model.",
        )
        parser.add_argument(
            "--sider_path",
            type=str,
            help="Optional path to SIDER model.",
        )
        return parent_parser

    def __init__(
        self,
        project_path,
        mol_model_path,
        protein_model_path,
        affinity_model_path,
        protein_data_path,
        params_path,
        test_protein_id,
        unbiased_predictions_path,
        tox21_path=None,
        organdb_path=None,
        site=None,
        clintox_path=None,
        sider_path=None,
        **kwargs,
    ):
        super(Reinforce_Base, self).__init__()
        # Default setting
        self.project_path = project_path
        # Read the parameters json file
        self.params = dict()
        with open(params_path) as f:
            self.params.update(json.load(f))
        # Set basic parameters
        self.epochs = self.params.get("epochs", 100)
        self.lr = self.params.get("learning_rate", 5e-06)
        self.opt_fn = OPTIMIZER_FACTORY[self.params.get("optimizer", "Adam")]
        self.batch_size = self.params.get("batch_size", 128)
        self.generate_len = self.params.get("generate_len", 100)
        self.temperature = self.params.get("temperature", 1.4)
        # Passing optional paths to params to possibly update_reward_fn
        optional_rewards = [
            (tox21_path, "tox21_path"),
            (organdb_path, "organdb_path"),
            (site, "site_path"),
            (clintox_path, "clintox_path"),
            (sider_path, "sider_path"),
        ]
        for (args, kwargs) in optional_rewards:
            if args:
                # json still has presedence
                self.params[kwargs] = project_path + args
        # Restore ProtVAE model
        protein_model = ProtVAE_Module.load_from_checkpoint(
            os.path.join(
                project_path + protein_model_path, "ProtVAE.ckpt"
            ),
            params_filepath="Config/ProtVAE.json",
        )
        self.encoder = protein_model.encoder
        # Restore ChemVAE model (only use decoder)
        chemistry_model = ChemVAE_Module.load_from_checkpoint(
            os.path.join(
                project_path + mol_model_path, "ChemVAE.ckpt"
            ),
            project_filepath=project_path,
            params_filepath="Config/ChemVAE.json",
            smiles_language_filepath="Config/selfies_language.pkl",
        )
        self.decoder = chemistry_model.decoder
        # Load smiles languages for decoder
        decoder_smiles_language = SMILESLanguage.load(
            os.path.join(project_path, "Config/selfies_language.pkl")
        )
        self.decoder._associate_language(decoder_smiles_language)
        # Restore affinity predictor
        self.predictor = Predictor_Module.load_from_checkpoint(
            os.path.join(
                project_path + affinity_model_path, "Predictor.ckpt"
            ),
            params_filepath="Config/Predictor.json",
        )
        # Load smiles and protein languages for predictor
        predictor_smiles_language = SMILESLanguage.load(
            os.path.join(project_path, "Config/smiles_language.pkl")
        )
        predictor_protein_language = ProteinLanguage.load(
            os.path.join(project_path, "Config/protein_language.pkl")
        )
        self.predictor._associate_language(predictor_smiles_language)
        self.predictor._associate_language(predictor_protein_language)
        # Set padding parameters
        self.pad_smiles_predictor = LeftPadding(
            self.predictor.smiles_padding_length,
            self.predictor.smiles_language.padding_index,
        )
        self.pad_protein_predictor = LeftPadding(
            self.predictor.protein_padding_length,
            self.predictor.protein_language.padding_index,
        )
        # Load protein sequence data for protein test name
        protein_dataset = ProteinDataset(
            protein_data_path, test_protein_id
        )
        self.protein_df = protein_dataset.origin_protein_df
        # Specifies the baseline model used for comparison
        self.protein_test_name = self.protein_df.iloc[test_protein_id].name
        self.unbiased_preds = np.array(
            pd.read_csv(
                os.path.join(
                    unbiased_predictions_path,
                    self.protein_test_name + ".csv",
                )
            )["affinity"].values
        )

    def forward(self, protein_name):
        # Set evaluate mode for encoder, predictor and drug evaluator
        self.encoder.eval()
        self.predictor.eval()
        drug_evaluator = [
            self.tox21,
            self.organdb,
            self.clintox,
            self.sider,
        ]
        for evaluator in drug_evaluator:
            if evaluator is not None:
                evaluator.model.eval()

    def training_step(self, batch, *args, **kwargs):
        return NotImplementedError

    def training_epoch_end(self, outputs):
        return NotImplementedError

    def configure_optimizers(self):
        opt = self.opt_fn(self.decoder.parameters(), lr=self.lr)
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

    """
    -------------------------
    Reinforce learning method
    -------------------------
    """

    def update_params(self, params):
        # Parameters for reward function
        self.qed = QED()
        self.scscore = SCScore()
        self.esol = ESOL()
        self.sas = SAS()
        self.qed_weight = params.get("qed_weight", 0.0)
        self.scscore_weight = params.get("scscore_weight", 0.0)
        self.esol_weight = params.get("esol_weight", 0.0)
        self.tox21_weight = params.get("tox21_weight", 0.5)
        if self.tox21_weight > 0.0:
            self.tox21 = Tox21(
                project_path=self.project_path,
                params_path="Config/Toxicity.json",
                model_path=params.get(
                    "tox21_path", os.path.join("..", "data", "models", "Tox21")
                )
                + "Toxicity.ckpt",
                device=self.device,
                reward_type="raw",
            )
            self.tox21.model.to(self.device)
        else:
            self.tox21 = None
        self.organdb_weight = params.get("organdb_weight", 0.0)
        if self.organdb_weight > 0.0:
            self.organdb = OrganDB(
                self.project_path,
                params_path="Config/Toxicity.json",
                model_path=params.get(
                    "organdb_path", os.path.join("..", "data", "models", "OrganDB")
                ),
                site=params["site"],
                device=self.device,
            )
            self.organdb.model.to(self.device)
        else:
            self.organdb = None
        self.clintox_weight = params.get("clintox_weight", 0.0)
        if self.clintox_weight > 0.0:
            self.clintox = ClinTox(
                project_path=self.project_path,
                params_path="Config/Toxicity.json",
                model_path=params.get(
                    "clintox_path", os.path.join("..", "data", "models", "ClinTox")
                ),
                device=self.device,
            )
            self.clintox.model.to(self.device)
        else:
            self.clintox = None
        self.sider_weight = params.get("sider_weight", 0.0)
        if self.sider_weight > 0.0:
            self.sider = SIDER(
                self.project_path,
                params_path="Config/Toxicity.json",
                model_path=params.get(
                    "sider_path", os.path.join("..", "data", "models", "Siders")
                ),
                device=self.device,
            )
            self.sider.model.to(self.device)
        else:
            self.sider = None
        self.affinity_weight = params.get("affinity_weight", 1.0)

        def tox_f(s):
            x = 0
            if self.tox21_weight > 0.0:
                x += self.tox21_weight * self.tox21(s)
            if self.sider_weight > 0.0:
                x += self.sider_weight * self.sider(s)
            if self.clintox_weight > 0.0:
                x += self.clintox_weight * self.clintox(s)
            if self.organdb_weight > 0.0:
                x += self.organdb_weight * self.organdb(s)
            return x

        # This is the joint reward function. Each score is normalized to be
        # inside the range [0, 1].
        self.reward_fn = lambda smiles, protein: (
            self.affinity_weight * self.get_reward_affinity(smiles, protein)
            + th.Tensor([tox_f(s) for s in smiles]).to(self.device)
        )
        # discount factor
        self.gamma = params.get("gamma", 0.99)
        # maximal length of generated molecules
        self.generate_len = params.get(
            "generate_len", self.predictor.params["smiles_padding_length"] - 2
        )
        # smoothing factor for softmax during token sampling in decoder
        self.temperature = params.get("temperature", 0.8)
        # gradient clipping in decoder
        self.grad_clipping = params.get("clip_grad", None)

    def encode_protein(self, protein=None, batch_size=128):
        """
        Encodes protein in latent space with protein encoder.
        Args:
            protein (str): Name of a protein
            batch_size (int): batch_size
        """
        if protein is None:
            latent_z = th.randn(1, batch_size, self.encoder.latent_size)
        else:

            protein_tensor, _ = self.protein_to_numerical(
                protein, encoder_uses_sequence=False
            )
            protein_mu, protein_logvar = self.encoder(protein_tensor)

            latent_z = th.unsqueeze(
                # Reparameterize
                th.randn_like(protein_mu.repeat(batch_size, 1))
                .mul_(th.exp(0.5 * protein_logvar.repeat(batch_size, 1)))
                .add_(protein_mu.repeat(batch_size, 1)),
                0,
            )
        latent_z = latent_z.to(self.device)

        return latent_z

    def smiles_to_numerical(self, smiles_list, target="predictor"):
        """
        Receives a list of SMILES.
        Converts it to a numerical torch Tensor according to smiles_language
        """

        if target == "generator":
            # NOTE: Code for this in the normal REINFORCE class
            raise ValueError("Priming drugs not yet supported")
        smiles_to_tensor = ToTensor(self.device)
        smiles_num = [
            th.unsqueeze(
                smiles_to_tensor(
                    self.pad_smiles_predictor(
                        self.predictor.smiles_language.smiles_to_token_indexes(smiles)
                    )
                ),
                0,
            ).to(self.device)
            for smiles in smiles_list
        ]
        # Catch scenario where all SMILES are invalid
        smiles_tensor = th.Tensor().to(self.device)
        if len(smiles_num) > 0:
            smiles_tensor = th.cat(smiles_num, dim=0).to(self.device)
        return smiles_tensor

    def protein_to_numerical(
        self, protein, encoder_uses_sequence=True, predictor_uses_sequence=True
    ):
        """
        Receives a name of a protein.
        Returns two numerical torch Tensor, the first for the protein encoder,
        the second for the affinity predictor.
        Args:
            protein (str): Name of the protein
            encoder_uses_sequence (bool): Whether the encoder uses the protein
                sequence or an embedding.
            predictor_uses_sequence (bool): Whether the predictor uses the
                protein sequence or an embedding.

        """
        protein_to_tensor = ToTensor(self.device)
        protein_sequence = self.protein_df.loc[protein]["Sequence"]
        if predictor_uses_sequence:
            sequence_tensor_p = th.unsqueeze(
                protein_to_tensor(
                    self.pad_protein_predictor(
                        self.predictor.protein_language.sequence_to_token_indexes(
                            protein_sequence
                        )
                    )
                ),
                0,
            ).to(self.device)
        if encoder_uses_sequence:
            sequence_tensor_e = th.unsqueeze(
                protein_to_tensor(
                    self.pad_protein_predictor(
                        self.encoder.protein_language.sequence_to_token_indexes(
                            protein_sequence
                        )
                    )
                ),
                0,
            ).to(self.device)
        if (not encoder_uses_sequence) or (not predictor_uses_sequence):
            # Column names of DF
            locations = [str(x) for x in range(768)]
            protein_encoding = self.protein_df.loc[protein][locations]
            encoding_tensor = th.unsqueeze(th.Tensor(protein_encoding), 0).to(
                self.device
            )
        t1 = sequence_tensor_e if encoder_uses_sequence else encoding_tensor
        t2 = sequence_tensor_p if predictor_uses_sequence else encoding_tensor
        return t1, t2

    def get_smiles_from_latent(self, latent, remove_invalid=True):
        """
        Takes some samples from latent space.
        Args:
            latent (torch.Tensor): tensor of shape 1 x batch_size x latent_dim.
            remove_invalid (bool): whether invalid SMILES are to be removed.
                Deaults to True.

        Returns:
            tuple(list, list): SMILES and numericals.
        """
        if self.decoder.latent_dim == 2 * self.encoder.latent_size:
            latent = latent.repeat(1, 1, 2)
        mols_numerical = self.decoder.generate(
            latent,
            prime_input=th.Tensor([self.decoder.smiles_language.start_index])
            .long()
            .to(self.device),
            end_token=th.Tensor([self.decoder.smiles_language.stop_index])
            .long()
            .to(self.device),
            generate_len=self.generate_len,
            search=SamplingSearch(temperature=self.temperature),
        )
        # Retrieve SMILES from numericals
        smiles_num_tuple = [
            (
                self.decoder.smiles_language.token_indexes_to_smiles(mol_num.tolist()),
                th.cat(
                    [
                        mol_num.long().to(self.device),
                        th.tensor(2 * [self.decoder.smiles_language.stop_index]).to(
                            self.device
                        ),
                    ]
                ).to(self.device),
            )
            for mol_num in iter(mols_numerical)
        ]
        numericals = [sm[1] for sm in smiles_num_tuple]

        # NOTE: If SMILES is used instead of SELFIES this line needs adjustment
        smiles = [
            self.decoder.smiles_language.selfies_to_smiles(sm[0])
            for sm in smiles_num_tuple
        ]
        imgs = [
            Chem.MolFromSmiles(s, sanitize=True)
            if type(s) == str and len(s) > 5
            else None
            for s in smiles
        ]
        valid_idxs = [ind for ind in range(len(imgs)) if imgs[ind] is not None]
        smiles = [
            smiles[ind]
            for ind in range(len(imgs))
            if not (remove_invalid and imgs[ind] is None)
        ]
        nums = [
            numericals[ind]
            for ind in range(len(numericals))
            if not (remove_invalid and imgs[ind] is None)
        ]
        self.log(
            "SMILES validity (%)",
            (len([i for i in imgs if i is not None]) / len(imgs)) * 100,
        )
        return smiles, nums, valid_idxs

    def generate_compounds_and_evaluate(
        self,
        batch_size,
        protein=None,
        primed_drug=" ",
        return_latent=False,
        remove_invalid=True,
    ):
        """
        Generate some compounds and evaluate them with the predictor

        Args:
            epoch (int): The training epoch.
            batch_size (int): The batch size.
            protein (str): A string, the protein used to drive generator.
            primed_drug (str): SMILES string to prime the generator.

        Returns:
            np.array: Predictions from PaccMann.
        """
        if primed_drug != " ":
            raise ValueError("Drug priming not yet supported.")
        if protein is None:
            # Generate a random molecule
            latent_z = th.randn(1, batch_size, self.decoder.latent_dim)
        else:
            (
                protein_encoder_tensor,
                protein_predictor_tensor,
            ) = self.protein_to_numerical(protein, encoder_uses_sequence=False)
            protein_mu, protein_logvar = self.encoder(protein_encoder_tensor)
            latent_z = th.unsqueeze(
                # Reparameterize
                th.rand_like(protein_mu.repeat(batch_size, 1))
                .mul_(th.exp(0.5 * protein_logvar.repeat(batch_size, 1)))
                .add_(protein_mu.repeat(batch_size, 1)),
                0,
            )
        latent_z = latent_z.to(self.device)
        # Generate drugs
        valid_smiles, valid_nums, _ = self.get_smiles_from_latent(
            latent_z, remove_invalid=remove_invalid
        )
        smiles_t = self.smiles_to_numerical(valid_smiles, target="predictor")
        # Evaluate drugs
        pred, pred_dict = self.predictor(
            smiles_t, protein_predictor_tensor.repeat(len(valid_smiles), 1)
        )
        if return_latent:
            return valid_smiles, pred.detach().squeeze(), latent_z
        else:
            return valid_smiles, pred

    def update_reward_fn(self, params):
        """Set the reward function
        Arguments:
            params (dict): Hyperparameter for PaccMann reward function
        """
        self.affinity_weight = self.params.get("affinity_weight", 1.0)

        def tox_f(s):
            x = 0
            if self.tox21_weight > 0.0:
                x += self.tox21_weight * self.tox21(s)
            if self.sider_weight > 0.0:
                x += self.sider_weight * self.sider(s)
            if self.clintox_weight > 0.0:
                x += self.clintox_weight * self.clintox(s)
            if self.organdb_weight > 0.0:
                x += self.organdb_weight * self.organdb(s)
            return x

        # This is the joint reward function. Each score is normalized to be
        # inside the range [0, 1].
        self.reward_fn = lambda smiles, protein: (
            self.affinity_weight * self.get_reward_affinity(smiles, protein)
            + th.Tensor([tox_f(s) for s in smiles]).to(self.device)
        )

    def get_reward_affinity(self, valid_smiles, protein):
        """
        Get the reward from affinity predictor

        Args:
            valid_smiles (list): A list of valid SMILES strings.
            protein (str): Name of target protein

        Returns:
            np.array: computed reward (fixed to 1/(1+exp(x))).
        """
        # Build up SMILES tensor and GEP tensor
        smiles_tensor = self.smiles_to_numerical(valid_smiles, target="predictor")
        # If all SMILES are invalid, no reward is given
        if len(smiles_tensor) == 0:
            return 0

        _, protein_tensor = self.protein_to_numerical(
            protein, encoder_uses_sequence=False
        )
        pred, pred_dict = self.predictor(
            smiles_tensor, protein_tensor.repeat(smiles_tensor.shape[0], 1)
        )
        return pred.detach().squeeze()
