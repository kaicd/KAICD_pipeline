import argparse
import json
import os
import sys
import logging

import numpy as np
import pandas as pd
from PIL import Image as pilimg
from torch.optim import Adam
import pytorch_lightning as pl
from wandb import Image
from wandb import Table
from rdkit import Chem
from rdkit.Chem import Draw
from paccmann_omics.encoders import ENCODER_FACTORY
from paccmann_chemistry.models import StackGRUDecoder, StackGRUEncoder, TeacherVAE
from paccmann_chemistry.utils import get_device
from paccmann_generator import ReinforceProtein
from paccmann_generator.plot_utils import plot_and_compare_proteins
from paccmann_predictor.models import MODEL_FACTORY
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage

from datasets.ProteinDataset import ProteinDataset


class Reinforce_lightning(pl.LightningModule):
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
            default="models/SELFIESVAE",
        )
        parser.add_argument(
            "--protein_model_path",
            type=str,
            help="Path to protein model",
            default="models/ProteinVAE",
        )
        parser.add_argument(
            "--affinity_model_path",
            type=str,
            help="Path to pretrained affinity model",
            default="models/affinity",
        )
        parser.add_argument(
            "--params_path",
            type=str,
            help="Model params json file directory",
            default="code/paccmann_generator/examples/affinity/conditional_generator.json",
        )
        parser.add_argument(
            "--model_name",
            type=str,
            help="Name for the trained model.",
            default="paccmann_sarscov2",
        )
        parser.add_argument(
            "--unbiased_predictions_path",
            type=str,
            help="Path to folder with aff. preds for 3000 mols from unbiased generator",
            default="data/training/unbiased_predictions",
        )
        parser.add_argument(
            "--save_interval", type=int, help="Sets model storage interval.", default=10
        )
        parser.add_argument(
            "--fig_save_path",
            type=str,
            help="Path to save result figure.",
            default="/home/lhs/PaccMann_Lightning/binding_image/",
        )
        parser.add_argument(
            "--tox21_path", help="Optional path to Tox21 model.", default="/home/lhs/paccmann_sarscov2/models/Tox21"
        )
        parser.add_argument("--organdb_path", help="Optional path to OrganDB model.", default="/home/lhs/paccmann_sarscov2/models/OrganDB")
        parser.add_argument(
            "--site", help="Specify a site in case of using a OrganDB model.", default="/home/lhs/paccmann_sarscov2/models/site"
        )
        parser.add_argument("--clintox_path", help="Optional path to ClinTox model.", default="/home/lhs/paccmann_sarscov2/models/ClinTox")
        parser.add_argument("--sider_path", help="Optional path to SIDER model.", default="/home/lhs/paccmann_sarscov2/models/SIDER")

        return parent_parser

    def __init__(
        self,
        project_path,
        mol_model_path,
        protein_model_path,
        affinity_model_path,
        protein_data_path,
        params_path,
        model_name,
        test_protein_id,
        unbiased_predictions_path,
        fig_save_path,
        save_interval=10,
        tox21_path=None,
        organdb_path=None,
        site=None,
        clintox_path=None,
        sider_path=None,
        **kwargs
    ):
        super(Reinforce_lightning, self).__init__()

        # Default setting
        self.save_interval = save_interval
        self.fig_save_path = fig_save_path
        self.automatic_optimization = False

        # Setup logging
        logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
        logger = logging.getLogger("train_paccmann_rl")
        logger_m = logging.getLogger("matplotlib")
        logger_m.setLevel(logging.WARNING)

        # Read the params json
        self.params = dict()
        with open(project_path + params_path) as f:
            self.params.update(json.load(f))

        # Passing optional paths to params to possibly update_reward_fn
        optional_reward_args = [
            tox21_path,
            organdb_path,
            site,
            clintox_path,
            sider_path,
        ]
        optional_reward_kwargs = [
            "tox21_path",
            "organdb_path",
            "site",
            "clintox_path",
            "sider_path",
        ]
        for kwargs, arg in zip(optional_reward_kwargs, optional_reward_args):
            if arg:
                # json still has presedence
                self.params[kwargs] = arg

        # Restore SMILES Model
        with open(
            os.path.join(project_path + mol_model_path, "model_params.json")
        ) as f:
            mol_params = json.load(f)
        encoder = StackGRUEncoder(mol_params)
        decoder = StackGRUDecoder(mol_params)
        self.generator = TeacherVAE(encoder, decoder)
        self.generator.load(
            os.path.join(
                project_path + mol_model_path,
                f"weights/best_{self.params.get('metric', 'rec')}.pt",
            ),
            map_location=get_device(),
        )
        self.generator.encoder.eval()

        # Load smiles languages for generator
        generator_smiles_language = SMILESLanguage.load(
            os.path.join(project_path + mol_model_path, "selfies_language.pkl")
        )
        self.generator._associate_language(generator_smiles_language)

        # Restore protein model
        with open(
            os.path.join(project_path + protein_model_path, "model_params.json")
        ) as f:
            protein_params = json.load(f)
        protein_encoder = ENCODER_FACTORY["dense"](protein_params)
        protein_encoder.load(
            os.path.join(
                project_path + protein_model_path,
                f"weights/best_{self.params.get('metric', 'both')}_encoder.pt",
            ),
            map_location=get_device(),
        )
        protein_encoder.eval()

        # Restore affinity predictor
        with open(
            os.path.join(project_path + affinity_model_path, "model_params.json")
        ) as f:
            predictor_params = json.load(f)
        predictor = MODEL_FACTORY["bimodal_mca"](predictor_params)
        predictor.load(
            os.path.join(
                project_path + affinity_model_path,
                f"weights/best_{self.params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt",
            ),
            map_location=get_device(),
        )
        predictor.eval()

        # Load smiles and protein languages for predictor
        affinity_smiles_language = SMILESLanguage.load(
            os.path.join(project_path + affinity_model_path, "smiles_language.pkl")
        )
        affinity_protein_language = ProteinLanguage.load(
            os.path.join(project_path + affinity_model_path, "protein_language.pkl")
        )
        predictor._associate_language(affinity_smiles_language)
        predictor._associate_language(affinity_protein_language)

        # Load protein sequence data for protein test name
        protein_dataset = ProteinDataset(project_path + protein_data_path, test_protein_id)
        protein_df = protein_dataset.origin_protein_df

        # Specifies the baseline model used for comparison
        self.protein_test_name = protein_df.iloc[test_protein_id].name
        self.unbiased_preds = np.array(
            pd.read_csv(
                os.path.join(
                    project_path + unbiased_predictions_path, self.protein_test_name + ".csv"
                )
            )["affinity"].values
        )

        # Define reinforcement learning model
        model_folder_name = model_name
        self.learner = ReinforceProtein(
            self.generator,
            protein_encoder,
            predictor,
            protein_df,
            self.params,
            model_folder_name,
            logger,
        )

        # Define for save result(good molecules!) in dataframe
        self.biased_ratios, self.tox_ratios = [], []
        self.rewards, self.rl_losses = [], []
        self.gen_mols, self.gen_prot, self.gen_affinity, self.mode = [], [], [], []

    def forward(self, protein_name):
        rew, loss = self.learner.policy_gradient(
            protein_name, self.current_epoch, self.params["batch_size"]
        )

        return rew, loss

    def training_step(self, batch, *args, **kwargs):
        rew, loss = self(batch[0])

        self.log("mean_reward", rew.item())
        self.log("rl_loss", loss)

        return {"mean_rewards": rew, "rl_loss": loss}

    def training_epoch_end(self, outputs):
        for o in outputs:
            self.rewards.append(o["mean_rewards"].item())
            self.rl_losses.append(o["rl_loss"])

        if self.current_epoch % self.save_interval == 0:
            self.learner.save(f"gen_{self.current_epoch}.pt", f"enc_{self.current_epoch}.pt")

        smiles, preds = self.learner.generate_compounds_and_evaluate(
            self.current_epoch, self.params["eval_batch_size"], self.protein_test_name
        )
        toxes = np.array([self.learner.tox21(s) for s in smiles])

        # Filtering (affinity > 0.5)
        useful_smiles = [s for i, s in enumerate(smiles) if preds[i] > 0.5]
        useful_preds = preds[preds > 0.5]
        useful_toxes = [self.learner.tox21(s) for s in useful_smiles]

        for p, s in zip(useful_preds, useful_smiles):
            self.gen_mols.append(s)
            self.gen_prot.append(self.protein_test_name)
            self.gen_affinity.append(p)
            self.mode.append("eval")

        # Filtering (tox == 1.0 -> non-toxic)
        non_toxic_useful_smiles = [
            s for i, s in enumerate(useful_smiles) if useful_toxes[i] == 1.0
        ]
        non_toxic_useful_preds = useful_preds[useful_toxes == 1.0]

        # Log top 5 generate molecule
        lead = []
        for smiles in non_toxic_useful_smiles:
            mol = Chem.MolFromSmiles(smiles)
            if mol:
                lead.append(mol)
                if len(lead) == 5:
                    break

        if len(lead) > 0:
            self.logger.experiment.log({"Top N Generative Molecules": [Image(np.array(Draw.MolsToImage(lead)), caption="Good Molecules")]})

        # Log toxic ratio
        plot_and_compare_proteins(
            self.unbiased_preds,
            preds,
            self.protein_test_name,
            self.current_epoch,
            self.fig_save_path,
            "train",
            self.params["eval_batch_size"],
        )

        biased_ratio = np.round(100 * (np.sum(preds > 0.5) / len(preds)), 1)
        tox_ratio = np.round(100 * (np.sum(toxes == 1.0) / len(toxes)), 1)
        img = self.fig_save_path + f"train_{self.protein_test_name}_epoch_{self.current_epoch}_eff_{biased_ratio}.png"
        self.logger.experiment.log(
            {
                "NAIVE and BIASED binding compounds distribution": [Image(pilimg.open(img))]
            }
        )
        self.biased_ratios.append(biased_ratio)
        self.tox_ratios.append(tox_ratio)

        # Log total results(dataframe)
        df = Table(
            dataframe=pd.DataFrame(
                {
                    "protein": self.gen_prot,
                    "SMILES": self.gen_mols,
                    "Binding probability": self.gen_affinity,
                    "mode": self.mode,
                    "Tox21": useful_toxes,
                }
            )
        )
        self.log("Results", df)

    def configure_optimizers(self):
        opt = Adam(
            self.generator.decoder.parameters(),
            lr=self.params.get("learning_rate", 0.0001),
            eps=self.params.get("eps", 0.0001),
            weight_decay=self.params.get("weight_decay", 0.00001),
        )

        return [opt]
