from argparse import ArgumentParser

import json
import h5py
import torch as th
import pytorch_lightning as pl

from utils import HDFDataset, BlockDataLoader


class ChemGG_DataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_hdf_filepath: str,
        valid_hdf_filepath: str,
        test_hdf_filepath: str,
        params_filepath: str,
        *args,
        **kwargs,
    ) -> None:
        super(ChemGG_DataModule, self).__init__()
        self.train_hdf_filepath = train_hdf_filepath
        self.valid_hdf_filepath = valid_hdf_filepath
        self.test_hdf_filepath = test_hdf_filepath
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
            block_size=self.params.get("block_size", 10000),
            shuffle=self.params.get("shuffle", True),
            n_workers=self.params.get("n_workers", 32),
            pin_memory=self.params.get("pin_memory", True),
        )

    def train_dataloader(self) -> BlockDataLoader:
        return self.dataloader(self.train_dataset)

    def val_dataloader(self) -> BlockDataLoader:
        return self.dataloader(self.valid_dataset)

    def test_dataloader(self) -> BlockDataLoader:
        return self.dataloader(self.test_dataset)
