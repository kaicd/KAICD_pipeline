import json
import os.path as osp
from argparse import ArgumentParser
from typing import Optional, Union, Dict, List

import dill
import pytorch_lightning as pl
from pytoda.datasets._csv_eager_dataset import _CsvEagerDataset
from torch.utils.data.dataloader import DataLoader


class ProtVAE_DataModule(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargsdouble
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_protein_filepath",
            type=str,
            default="data/pretraining/ProtVAE/tape_encoded/train_representation.csv",
            help="Path to the training data (.csv).",
        )
        parser.add_argument(
            "--test_protein_filepath",
            type=str,
            default="data/pretraining/ProtVAE/tape_encoded/val_representation.csv",
            help="Path to the testing data (.csv).",
        )
        parser.add_argument(
            "--train_dataset_filepath",
            type=str,
            default="preprocessing/protVAE_train_dataset.pkl",
            help="Path to the preprocessed training data (.pkl).",
        )
        parser.add_argument(
            "--test_dataset_filepath",
            type=str,
            default="preprocessing/protVAE_test_dataset.pkl",
            help="Path to the preprocessed testing data (.pkl).",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath,
        train_protein_filepath,
        test_protein_filepath,
        train_dataset_filepath,
        test_dataset_filepath,
        params_filepath,
        *args,
        **kwargs,
    ):
        super(ProtVAE_DataModule, self).__init__()
        self.dataset_filepath = project_filepath + "preprocessing/"
        self.train_protein_filepath = project_filepath + train_protein_filepath
        self.test_protein_filepath = project_filepath + test_protein_filepath
        self.train_dataset_filepath = project_filepath + train_dataset_filepath
        self.test_dataset_filepath = project_filepath + test_dataset_filepath
        self.params_filepath = params_filepath
        self.params = {}

        # Process parameter file
        with open(self.params_filepath) as f:
            self.params.update(json.load(f))

    def setup(self, stage: Optional[str] = None) -> None:
        protein_filepath = [self.train_protein_filepath, self.test_protein_filepath]

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
            with open(self.train_dataset_filepath, "wb") as f:
                dill.dump(self.train_dataset, f)
            with open(self.test_dataset_filepath, "wb") as f:
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
