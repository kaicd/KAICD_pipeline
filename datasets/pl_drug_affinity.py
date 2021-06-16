import json
from argparse import ArgumentParser
import torch as th
import pytorch_lightning as pl
from typing import Optional, Union, Dict, List
from pytoda.datasets import DrugAffinityDataset
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage
from torch.utils.data.dataloader import DataLoader


class DrugAffinity_lightning(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_affinity_filepath",
            type=str,
            default="data/pretraining/affinity_predictor/filtered_train_binding_data.csv",
            help="Path to the drug affinity data.",
        )
        parser.add_argument(
            "--test_affinity_filepath",
            type=str,
            default="data/pretraining/affinity_predictor/filtered_val_binding_data.csv",
            help="Path to the drug affinity data.",
        )
        parser.add_argument(
            "--protein_filepath",
            type=str,
            default="data/pretraining/affinity_predictor/sequences.smi",
            help="Path to the protein profile data.",
        )
        parser.add_argument(
            "--smi_filepath",
            type=str,
            default="data/pretraining/affinity_predictor/filtered_ligands.smi",
            help="Path to the SMILES data.",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath,
        train_affinity_filepath,
        test_affinity_filepath,
        protein_filepath,
        smi_filepath,
        smiles_language_filepath,
        protein_language_filepath,
        params_filepath,
        device,
        *args,
        **kwargs,
    ):
        super(DrugAffinity_lightning, self).__init__()
        self.train_affinity_filepath = project_filepath + train_affinity_filepath
        self.test_affinity_filepath = project_filepath + test_affinity_filepath
        self.protein_filepath = project_filepath + protein_filepath
        self.smi_filepath = project_filepath + smi_filepath
        self.smiles_language_filepath = project_filepath + smiles_language_filepath
        self.protein_language_filepath = project_filepath + protein_language_filepath
        self.params_filepath = project_filepath + params_filepath
        self.smiles_language = SMILESLanguage.load(self.smiles_language_filepath)
        self.protein_language = ProteinLanguage.load(self.protein_language_filepath)
        self.params = {}
        self.device = device

        # Process parameter file
        with open(self.params_filepath) as f:
            self.params.update(json.load(f))

    def setup(self, stage: Optional[str] = None) -> None:
        drug_affinity_filepath = [
            self.train_affinity_filepath,
            self.test_affinity_filepath,
        ]
        smiles_augment = [self.params.get("augment_smiles", False), False]
        protein_augment_by_revert = [self.params.get("protein_augment", False), False]

        for i in range(2):
            dataset = DrugAffinityDataset(
                drug_affinity_filepath=drug_affinity_filepath[i],
                smi_filepath=self.smi_filepath,
                protein_filepath=self.protein_filepath,
                smiles_language=self.smiles_language,
                protein_language=self.protein_language,
                smiles_padding=self.params.get("smiles_padding", True),
                smiles_padding_length=self.params.get("smiles_padding_length", None),
                smiles_add_start_and_stop=self.params.get(
                    "smiles_add_start_stop", True
                ),
                smiles_augment=smiles_augment[i],
                smiles_canonical=self.params.get("smiles_canonical", False),
                smiles_kekulize=self.params.get("smiles_kekulize", False),
                smiles_all_bonds_explicit=self.params.get(
                    "smiles_bonds_explicit", False
                ),
                smiles_all_hs_explicit=self.params.get("smiles_all_hs_explicit", False),
                smiles_remove_bonddir=self.params.get("smiles_remove_bonddir", False),
                smiles_remove_chirality=self.params.get(
                    "smiles_remove_chirality", False
                ),
                smiles_selfies=self.params.get("selfies", False),
                protein_amino_acid_dict=self.params.get(
                    "protein_amino_acid_dict", "iupac"
                ),
                protein_padding=self.params.get("protein_padding", True),
                protein_padding_length=self.params.get("protein_padding_length", None),
                protein_add_start_and_stop=self.params.get(
                    "protein_add_start_stop", True
                ),
                protein_augment_by_revert=protein_augment_by_revert[i],
                drug_affinity_dtype=th.float,
                device=self.device,
                backend="eager",
            )

            if i == 0:
                self.train_dataset = dataset
            else:
                self.test_dataset = dataset

    def dataloader(self, dataset, shuffle, **kwargs):
        return DataLoader(
            dataset=dataset,
            batch_size=self.params["batch_size"],
            shuffle=shuffle,
            drop_last=True,
            num_workers=self.params.get("num_workers", 4),
        )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return self.dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader(self.test_dataset, shuffle=False)
