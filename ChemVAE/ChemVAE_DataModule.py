import json
import dill
import os.path as osp
from argparse import ArgumentParser

import pytorch_lightning as pl
from typing import Optional, Union, Dict, List
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from torch.utils.data.dataloader import DataLoader


class SELFIES_VAE_lightning(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
            cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_smiles_filepath",
            type=str,
            default="data/pretraining/SELFIESVAE/train_chembl_22_clean_1576904_sorted_std_final.smi",
            help="Path to the drug affinity data.",
        )
        parser.add_argument(
            "--test_smiles_filepath",
            type=str,
            default="data/pretraining/SELFIESVAE/test_chembl_22_clean_1576904_sorted_std_final.smi",
            help="Path to the drug affinity data.",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath,
        train_smiles_filepath,
        test_smiles_filepath,
        smiles_language_filepath,
        params_filepath,
        *args,
        **kwargs,
    ):
        super(SELFIES_VAE_lightning, self).__init__()
        self.dataset_filepath = project_filepath + "preprocessing/"
        self.train_smiles_filepath = project_filepath + train_smiles_filepath
        self.test_smiles_filepath = project_filepath + test_smiles_filepath
        self.smiles_language_filepath = project_filepath + smiles_language_filepath
        self.params_filepath = params_filepath
        self.smiles_language = SMILESLanguage.load(self.smiles_language_filepath)
        self.params = {}

        # Process parameter file
        with open(self.params_filepath) as f:
            self.params.update(json.load(f))

    def setup(self, stage: Optional[str] = None) -> None:
        smiles_filepath = [self.test_smiles_filepath, self.test_smiles_filepath]

        if osp.exists(
            self.dataset_filepath + "ChemVAE_train_dataset.pkl"
        ) and osp.exists(self.dataset_filepath + "ChemVAE_test_dataset.pkl"):
            print("Preprocessing file already exists!\nLoading...")

            with open(self.dataset_filepath + "ChemVAE_train_dataset.pkl", "rb") as f:
                self.train_dataset = dill.load(f)
            with open(self.dataset_filepath + "ChemVAE_test_dataset.pkl", "rb") as f:
                self.test_dataset = dill.load(f)
            print("Done...!")
        else:
            print("Data preprocessing...")
            for i in range(2):
                dataset = SMILESDataset(
                    smiles_filepath[i],
                    smiles_language=self.smiles_language,
                    padding=False,
                    selfies=self.params.get("selfies", False),
                    add_start_and_stop=self.params.get("add_start_stop_token", True),
                    augment=self.params.get("augment_smiles", False),
                    canonical=self.params.get("canonical", False),
                    kekulize=self.params.get("kekulize", False),
                    all_bonds_explicit=self.params.get("all_bonds_explicit", False),
                    all_hs_explicit=self.params.get("all_hs_explicit", False),
                    remove_bonddir=self.params.get("remove_bonddir", False),
                    remove_chirality=self.params.get("remove_chirality", False),
                    backend="lazy",
                )
                if i == 0:
                    self.train_dataset = dataset
                else:
                    self.test_dataset = dataset

            print("Saving...")
            with open(self.dataset_filepath + "ChemVAE_train_dataset.pkl", "wb") as f:
                dill.dump(self.train_dataset, f)
            with open(self.dataset_filepath + "ChemVAE_test_dataset.pkl", "wb") as f:
                dill.dump(self.test_dataset, f)
            print("Done...!")

    def dataloader(self, dataset, shuffle, **kwargs):
        return DataLoader(
            dataset=dataset,
            batch_size=self.params.get("batch_size", 64),
            collate_fn=self.collate_fn,
            shuffle=shuffle,
            drop_last=True,
            pin_memory=self.params.get("pin_memory", True),
            num_workers=self.params.get("num_workers", 8),
        )

    def collate_fn(self, batch):
        """
        Collate function for DataLoader.
        Note: to be used as collate_fn in torch.utils.data.DataLoader.
        Args:
            batch: Batch of sequences.
        Returns:
            Sorted batch from longest to shortest.
        """
        return [
            batch[index]
            for index in map(
                lambda t: t[0],
                sorted(enumerate(batch), key=lambda t: t[1].shape[0], reverse=True),
            )
        ]

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return self.dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader(self.test_dataset, shuffle=True)
