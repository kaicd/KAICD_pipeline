from argparse import ArgumentParser
from typing import Optional
import json

from torch.utils.data import DataLoader
import pytorch_lightning as pl

from Utility.data_utils import HDFDataset, BlockDataLoader


class ChemGG_DataModule(pl.LightningDataModule):
    @classmethod
    def add_argparse_args(
        cls, parent_parser: ArgumentParser, **kwargs
    ) -> ArgumentParser:
        parser = parent_parser.add_argument_group(cls.__name__)
        parser.add_argument(
            "--train_hdf_filepath",
            type=str,
            default="data/pretraining/ChemGG/train.h5",
        )
        parser.add_argument(
            "--valid_hdf_filepath",
            type=str,
            default="data/pretraining/ChemGG/valid.h5",
        )
        parser.add_argument(
            "--test_hdf_filepath",
            type=str,
            default="data/pretraining/ChemGG/test.h5",
        )

        return parent_parser

    def __init__(
        self,
        project_filepath: str,
        train_hdf_filepath: str,
        valid_hdf_filepath: str,
        test_hdf_filepath: str,
        params_filepath: str,
        *args,
        **kwargs,
    ) -> None:
        super(ChemGG_DataModule, self).__init__()
        self.train_hdf_filepath = project_filepath + train_hdf_filepath
        self.valid_hdf_filepath = project_filepath + valid_hdf_filepath
        self.test_hdf_filepath = project_filepath + test_hdf_filepath
        # load parameters
        self.params = {}
        with open(params_filepath) as f:
            self.params.update(json.load(f))

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = HDFDataset(self.train_hdf_filepath)
        self.valid_dataset = HDFDataset(self.valid_hdf_filepath)
        self.test_dataset = HDFDataset(self.test_hdf_filepath)

    def dataloader(self, dataset: HDFDataset) -> BlockDataLoader:
        return BlockDataLoader(
            dataset=dataset,
            batch_size=self.params.get("batch_size", 1000),
            shuffle=self.params.get("shuffle", True),
            num_workers=self.params.get("n_workers", 32),
            pin_memory=self.params.get("pin_memory", True),
        )

    def train_dataloader(self) -> DataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> DataLoader:
        return self.dataloader(self.valid_dataset)

    def test_dataloader(self) -> DataLoader:
        return self.dataloader(self.test_dataset)
