"""PaccMann^RL: Policy gradient class"""
import json
import os
import pickle
import argparse

import dgl
import torch as th
import pandas as pd
import numpy as np
import pytorch_lightning as pl
from rdkit import Chem
from pytoda.transforms import LeftPadding, ToTensor
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage
from ogb.utils import smiles2graph

from Utility.data_utils import ProteinDataset
from Utility.utils import dgl_graph, get_fingerprints
from ChemVAE.ChemVAE_Module import ChemVAE_Module
from ProtVAE.ProtVAE_Module import ProtVAE_Module
from Predictor.PredictorBA_Module import PredictorBA_Module
from Predictor.PredictorEFA_Module import PredictorEFA_Module
from Utility.drug_evaluators import (
    AromaticRing,
    QED,
    SCScore,
    ESOL,
    SAS,
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
            "--chem_model_path",
            type=str,
            help="Path to chemistry model",
            default="ChemVAE/checkpoint/",
        )
        parser.add_argument(
            "--prot_model_path",
            type=str,
            help="Path to protein model",
            default="ProtVAE/checkpoint/",
        )
        parser.add_argument(
            "--pred_model_path",
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
            "--chem_model_params_path",
            type=str,
            help="Chemistry model params json file directory",
            default="Config/ChemVAE.json",
        )
        parser.add_argument(
            "--prot_model_params_path",
            type=str,
            help="Protein model params json file directory",
            default="Config/ProtVAE.json",
        )
        parser.add_argument(
            "--pred_model_params_path",
            type=str,
            help="Prediction model params json file directory",
            default="Config/ProdictorBA.json",
        )
        parser.add_argument(
            "--chem_smiles_language_path",
            type=str,
            help="Prediction model params json file directory",
            default="Config/ChemVAE_selfies_language.pkl",
        )
        parser.add_argument(
            "--pred_smiles_language_path",
            type=str,
            help="Prediction model params json file directory",
            default="Config/PredictorBA_smiles_language.pkl",
        )
        parser.add_argument(
            "--pred_protein_language_path",
            type=str,
            help="Prediction model params json file directory",
            default="Config/PredictorBA_protein_language.pkl",
        )
        parser.add_argument(
            "--predictor_model_name",
            type=str,
            default="EFA",
        )
        parser.add_argument(
            "--unbiased_predictions_path",
            type=str,
            help="Path to folder with aff. preds for 3000 mols from unbiased generator",
            default="data/training/unbiased_predictions",
        )
        parser.add_argument(
            "--merged_sequence_encoding_path",
            type=str,
            default="data/training/merged_sequence_encoding",
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
        parser.add_argument(
            "--result_filepath", type=str, default="/raid/KAICD_sarscov2"
        )

        return parent_parser

    def __init__(
        self,
        project_path,
        chem_model_path,
        prot_model_path,
        pred_model_path,
        protein_data_path,
        params_path,
        chem_model_params_path,
        prot_model_params_path,
        pred_model_params_path,
        chem_smiles_language_path,
        pred_smiles_language_path,
        pred_protein_language_path,
        test_protein_id,
        predictor_model_name,
        unbiased_predictions_path,
        merged_sequence_encoding_path,
        result_filepath,
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
        self.predictor_model_name = predictor_model_name
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
                self.params[kwargs] = args
        # Restore ProtVAE model
        protein_model = ProtVAE_Module.load_from_checkpoint(
            prot_model_path,
            params_filepath=prot_model_params_path,
        )
        self.encoder = protein_model.encoder
        # Restore ChemVAE model (only use decoder)
        chemistry_model = ChemVAE_Module.load_from_checkpoint(
            chem_model_path,
            project_filepath=project_path,
            params_filepath=chem_model_params_path,
            smiles_language_filepath=chem_smiles_language_path,
        )
        self.decoder = chemistry_model.decoder
        # Load smiles languages for decoder
        decoder_smiles_language = SMILESLanguage.load(
            chem_smiles_language_path,
        )
        self.decoder._associate_language(decoder_smiles_language)
        # Restore affinity predictor
        if self.predictor_model_name == "EFA":
            self.predictor = PredictorEFA_Module.load_from_checkpoint(pred_model_path)
        else:
            self.predictor = PredictorBA_Module.load_from_checkpoint(
                pred_model_path,
                params_filepath=pred_model_params_path,
            )
            # Load smiles and protein languages for predictor
            predictor_smiles_language = SMILESLanguage.load(pred_smiles_language_path)
            predictor_protein_language = ProteinLanguage.load(pred_protein_language_path)
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
            project_path + protein_data_path, test_protein_id
        )
        self.protein_df = protein_dataset.origin_protein_df
        prottrans_path = os.path.join(
            project_path + merged_sequence_encoding_path,
            "uniprot_covid-19_prottrans.pkl",
        )
        with open(prottrans_path, "rb") as f:
            self.prottrans_enc = pickle.load(f)
        # Specifies the baseline model used for comparison
        self.protein_test_name = self.protein_df.iloc[test_protein_id].name
        self.unbiased_preds = np.array(
            pd.read_csv(
                os.path.join(
                    project_path + unbiased_predictions_path,
                    self.protein_test_name + ".csv",
                )
            )["affinity"].values
        )
        self.result_filepath = result_filepath

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
        self.ring = AromaticRing()
        self.qed = QED()
        self.scscore = SCScore()
        self.esol = ESOL()
        self.sas = SAS()
        self.ring_weight = params.get("ring_weight", 0.0)
        self.qed_weight = params.get("qed_weight", 0.0)
        self.scscore_weight = params.get("scscore_weight", 0.0)
        self.esol_weight = params.get("esol_weight", 0.0)
        self.sas_weight = params.get("sas_weight", 0.0)
        self.tox21_weight = params.get("tox21_weight", 0.5)
        if self.tox21_weight > 0.0:
            self.tox21 = Tox21(
                params_path="Config/Toxicity.json",
                model_path=params.get("tox21_path", ""),
                device=self.device,
                reward_type="raw",
            )
            self.tox21.model.to(self.device)
        else:
            self.tox21 = None
        self.organdb_weight = params.get("organdb_weight", 0.0)
        if self.organdb_weight > 0.0:
            self.organdb = OrganDB(
                params_path="Config/Toxicity.json",
                model_path=params.get("organdb_path", ""),
                site=params["site"],
                device=self.device,
            )
            self.organdb.model.to(self.device)
        else:
            self.organdb = None
        self.clintox_weight = params.get("clintox_weight", 0.0)
        if self.clintox_weight > 0.0:
            self.clintox = ClinTox(
                params_path="Config/Toxicity.json",
                model_path=params.get("clintox_path", ""),
                device=self.device,
            )
            self.clintox.model.to(self.device)
        else:
            self.clintox = None
        self.sider_weight = params.get("sider_weight", 0.0)
        if self.sider_weight > 0.0:
            self.sider = SIDER(
                params_path="Config/Toxicity.json",
                model_path=params.get("sider_path", ""),
                device=self.device,
            )
            self.sider.model.to(self.device)
        else:
            self.sider = None
        self.affinity_weight = params.get("affinity_weight", 1.0)

        def tox_f(s):
            x = 0
            if self.ring_weight > 0.0:
                x += self.ring_weight * self.ring(s)
            if self.qed_weight > 0.0:
                x += self.qed_weight * self.qed(s)
            if self.scscore_weight > 0.0:
                x += self.scscore_weight * self.scscore(s)
            if self.esol_weight > 0.0:
                x += self.esol_weight * self.esol(s)
            if self.sas_weight > 0.0:
                x += self.sas_weight * self.sas(s)
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
        self.predictor_fw = (
            self.get_efa_pred
            if self.predictor_model_name == "EFA"
            else self.get_ba_pred
        )

        self.reward_fn = lambda smiles, protein: (
            self.affinity_weight * self.predictor_fw(smiles, protein, score=True)
            + th.Tensor([tox_f(s) for s in smiles]).to(self.device)
        )
        # discount factor
        self.gamma = params.get("gamma", 0.99)
        # maximal length of generated molecules
        self.generate_len = params.get("generate_len", 100)
        # smoothing factor for softmax during token sampling in decoder
        self.temperature = params.get("temperature", 0.8)
        # gradient clipping in decoder
        self.grad_clipping = params.get("clip_grad", None)

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
            self.affinity_weight * self.predictor_fw(smiles, protein, score=True)
            + th.Tensor([tox_f(s) for s in smiles]).to(self.device)
        )

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

            protein_tensor = self.enc_protein_to_numerical(protein)
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

    def enc_protein_to_numerical(self, protein):
        locations = [str(x) for x in range(768)]
        protein_encoding = self.protein_df.loc[protein][locations]
        encoding_tensor = th.unsqueeze(th.Tensor(protein_encoding), 0).to(
            self.device
        )
        return encoding_tensor

    def pred_protein_to_numerical(self, protein):
        protein_to_tensor = ToTensor(self.device)
        protein_sequence = self.protein_df.loc[protein]["Sequence"]
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
        return sequence_tensor_p

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
            latent_z = th.randn(1, batch_size, self.decoder.latent_dim).to(self.device)
        else:
            protein_encoder_tensor = self.enc_protein_to_numerical(protein)
            protein_mu, protein_logvar = self.encoder(protein_encoder_tensor)
            latent_z = th.unsqueeze(
                # Reparameterize
                th.rand_like(protein_mu.repeat(batch_size, 1))
                .mul_(th.exp(0.5 * protein_logvar.repeat(batch_size, 1)))
                .add_(protein_mu.repeat(batch_size, 1)),
                0,
            ).to(self.device)
        # Generate drugs
        valid_smiles, valid_nums, _ = self.get_smiles_from_latent(
            latent_z, remove_invalid=remove_invalid
        )
        # Evaluate drugs
        pred = self.predictor_fw(valid_smiles, protein)

        if return_latent:
            return valid_smiles, pred.detach().squeeze(), latent_z
        else:
            return valid_smiles, pred

    def get_ba_pred(self, valid_smiles, protein, score=False):
        if len(valid_smiles) == 0:
            return 0

        smiles_tensor = self.smiles_to_numerical(valid_smiles, target="predictor")
        protein_tensor = self.pred_protein_to_numerical(protein)
        pred, pred_dict = self.predictor(
            smiles_tensor, protein_tensor.repeat(smiles_tensor.shape[0], 1)
        )
        return pred.detach().sequeeze() if score else pred

    def get_efa_pred(self, valid_smiles, protein, score=False):
        if len(valid_smiles) == 0:
            return 0

        g = dgl.batch([dgl_graph(smiles2graph(smiles)) for smiles in valid_smiles]).to(self.device)
        fp = th.as_tensor(get_fingerprints(valid_smiles), dtype=th.float32, device=self.device)
        pt = th.as_tensor(
            self.prottrans_enc[protein], dtype=th.float32, device=self.device
        ).repeat(len(valid_smiles), 1)

        pred = self.predictorEFA(g, fp, pt)
        return 1 / (1 + th.exp(pred.detach().sequeeze())) if score else pred