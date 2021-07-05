import json
import dill
import os.path as osp
from argparse import ArgumentParser
from typing import Optional, Union, Dict, List

import pytorch_lightning as pl
from torch.utils.data.dataloader import DataLoader
from pytoda.datasets._csv_eager_dataset import _CsvEagerDataset


class Protein_VAE_lightning(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargsdouble
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_filepath",
            type=str,
            default="data/pretraining/ProteinVAE/tape_encoded/train_representation.csv",
            help="Path to the training data (.csv).",
        )
        parser.add_argument(
            "--test_filepath",
            type=str,
            default="data/pretraining/ProteinVAE/tape_encoded/val_representation.csv",
            help="Path to the testing data (.csv).",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath,
        train_filepath,
        test_filepath,
        params_filepath,
        *args,
        **kwargs,
    ):
        super(Protein_VAE_lightning, self).__init__()
        self.dataset_filepath = project_filepath + "preprocessing/"
        self.train_filepath = project_filepath + train_filepath
        self.test_filepath = project_filepath + test_filepath
        self.params_filepath = project_filepath + params_filepath
        self.params = {}

        # Process parameter file
        with open(self.params_filepath) as f:
            self.params.update(json.load(f))

    def setup(self, stage: Optional[str] = None) -> None:
        protein_filepath = [self.train_filepath, self.test_filepath]

        if osp.exists(
            self.dataset_filepath + "protVAE_train_dataset.pkl"
        ) and osp.exists(self.dataset_filepath + "protVAE_test_dataset.pkl"):
            print("Preprocessing file already exists!\nLoading...")

            with open(self.dataset_filepath + "protVAE_train_dataset.pkl", "rb") as f:
                self.train_dataset = dill.load(f)
            with open(self.dataset_filepath + "protVAE_test_dataset.pkl", "rb") as f:
                self.test_dataset = dill.load(f)
            print("Done...!")
        else:
            print("Data preprocessing...")
            for i in range(2):
                dataset = _CsvEagerDataset(
                    protein_filepath[i],
                    feature_list=None,
                    index_col=0,
                    header=None,
                )
                if i == 0:
                    self.train_dataset = dataset
                else:
                    self.test_dataset = dataset

            print("Saving...")
            with open(self.dataset_filepath + "protVAE_train_dataset.pkl", "wb") as f:
                dill.dump(self.train_dataset, f)
            with open(self.dataset_filepath + "protVAE_test_dataset.pkl", "wb") as f:
                dill.dump(self.test_dataset, f)
            print("Done...!")

    def dataloader(self, dataset, shuffle, **kwargs):
        return DataLoader(
            dataset=dataset,
            batch_size=self.params.get("batch_size", 64),
            shuffle=shuffle,
        )

    def train_dataloader(
        self,
    ) -> Union[DataLoader, List[DataLoader], Dict[str, DataLoader]]:
        return self.dataloader(self.train_dataset, shuffle=True)

    def val_dataloader(self) -> Union[DataLoader, List[DataLoader]]:
        return self.dataloader(self.test_dataset, shuffle=True)
