import json
import os.path as osp
from argparse import ArgumentParser
from typing import Optional, Union, Dict, List

import dill
import pytorch_lightning as pl
from pytoda.datasets import SMILESDataset
from pytoda.smiles.smiles_language import SMILESLanguage
from torch.utils.data.dataloader import DataLoader


class ChemVAE_DataModule(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_smiles_filepath",
            type=str,
            default="data/pretraining/ChemVAE/train_chembl_22_clean_1576904_sorted_std_final.smi",
        )
        parser.add_argument(
            "--test_smiles_filepath",
            type=str,
            default="data/pretraining/ChemVAE/test_chembl_22_clean_1576904_sorted_std_final.smi",
        )
        parser.add_argument(
            "--train_dataset_filepath",
            type=str,
            default="preprocessing/ChemVAE_train_dataset_antiviral.pkl",
        )
        parser.add_argument(
            "--test_dataset_filepath",
            type=str,
            default="preprocessing/ChemVAE_test_dataset_antiviral.pkl",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath,
        train_smiles_filepath,
        test_smiles_filepath,
        train_dataset_filepath,
        test_dataset_filepath,
        smiles_language_filepath,
        params_filepath,
        device,
        *args,
        **kwargs,
    ):
        super(ChemVAE_DataModule, self).__init__()
        self.train_smiles_filepath = project_filepath + train_smiles_filepath
        self.test_smiles_filepath = project_filepath + test_smiles_filepath
        self.train_dataset_filepath = project_filepath + train_dataset_filepath
        self.test_dataset_filepath = project_filepath + test_dataset_filepath
        self.smiles_language_filepath = smiles_language_filepath
        self.params_filepath = params_filepath
        self.smiles_language = SMILESLanguage.load(self.smiles_language_filepath)
        self.params = {}
        self.device = device

        # Process parameter file
        with open(self.params_filepath) as f:
            self.params.update(json.load(f))

    def setup(self, stage: Optional[str] = None) -> None:
        smiles_filepath = [self.train_smiles_filepath, self.test_smiles_filepath]

        if osp.exists(self.train_dataset_filepath) and osp.exists(
            self.test_dataset_filepath
        ):
            print("Preprocessing file already exists!\nLoading...")

            with open(self.train_dataset_filepath, "rb") as f:
                self.train_dataset = dill.load(f)
            with open(self.test_dataset_filepath, "rb") as f:
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
            with open(self.train_dataset_filepath, "wb") as f:
                dill.dump(self.train_dataset, f)
            with open(self.test_dataset_filepath, "wb") as f:
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
        return self.dataloader(self.test_dataset, shuffle=False)
