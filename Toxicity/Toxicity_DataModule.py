import json
from argparse import ArgumentParser

import pytorch_lightning as pl
from typing import Optional, Union, Dict, List
from pytoda.datasets import AnnotatedDataset, SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from torch.utils.data.dataloader import DataLoader


class Toxicity_DataModule(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_score_filepath",
            type=str,
            default="data/pretraining/Toxicity/tox21_train.csv",
            help="Path to the training toxicity scores(.csv)",
        )
        parser.add_argument(
            "--test_score_filepath",
            type=str,
            default="data/pretraining/Toxicity/tox21_test.csv",
            help="Path to the test toxicity scores(.csv)",
        )
        parser.add_argument(
            "--smi_filepath",
            type=str,
            default="data/pretraining/Toxicity/tox21.smi",
            help="Path to the SMILES data(.smi)",
        )
        parser.add_argument(
            "--smiles_language_filepath",
            type=str,
            default="Config/Toxicity_smiles_language.pkl",
            help="Path to a pickle object a SMILES language object",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath,
        train_score_filepath,
        test_score_filepath,
        smi_filepath,
        smiles_language_filepath,
        params_filepath,
        *args,
        **kwargs,
    ):
        super(Toxicity_DataModule, self).__init__()
        self.train_score_filepath = project_filepath + train_score_filepath
        self.test_score_filepath = project_filepath + test_score_filepath
        self.smi_filepath = project_filepath + smi_filepath
        self.smiles_language_filepath = smiles_language_filepath
        self.smiles_language = SMILESLanguage.load(self.smiles_language_filepath)
        self.params_filepath = params_filepath
        self.params = {}

        # Process parameter file
        with open(self.params_filepath) as f:
            self.params.update(json.load(f))

    def setup(self, stage: Optional[str] = None) -> None:
        score_filepath = [self.train_score_filepath, self.test_score_filepath]
        randomize = [self.params.get("randomize", False), False]
        sanitize = [
            self.params.get("sanitize", True),
            self.params.get("test_sanitize", False),
        ]

        for i in range(2):
            smiles_dataset = SMILESDataset(
                self.smi_filepath,
                smiles_language=self.smiles_language,
                padding_length=self.params.get("smiles_padding_length", None),
                padding=self.params.get("padd_smiles", True),
                add_start_and_stop=self.params.get("add_start_stop_token", True),
                augment=self.params.get("augment_smiles", False),
                canonical=self.params.get("canonical", False),
                kekulize=self.params.get("kekulize", False),
                all_bonds_explicit=self.params.get("all_bonds_explicit", False),
                all_hs_explicit=self.params.get("all_hs_explicit", False),
                randomize=randomize[i],
                remove_bonddir=self.params.get("remove_bonddir", False),
                remove_chirality=self.params.get("remove_chirality", False),
                selfies=self.params.get("selfies", False),
                sanitize=sanitize[i],
                device=th.device("cpu"),
                backend="eager",
            )
            dataset = AnnotatedDataset(
                annotations_filepath=score_filepath[i],
                dataset=smiles_dataset,
                device=self.device,
            )

            if i == 0:
                self.train_dataset = dataset
            else:
                self.test_dataset = dataset

            if self.params.get("uncertainty", True) and self.params.get(
                "augment_test_data", False
            ):
                raise ValueError(
                    "Epistemic uncertainty evaluation not supported if augmentation "
                    "is not enabled for test data."
                )

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
        return self.dataloader(self.test_dataset, shuffle=False, drop_last=False)
