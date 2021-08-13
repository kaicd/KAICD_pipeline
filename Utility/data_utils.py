"""Utilities functions."""
import csv
import h5py
from typing import Tuple
import pandas as pd
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from pytoda.files import read_smi


def load_ts_properties(csv_path):
    """
    Loads training set properties from CSV, specified by the `csv_path`, and returns them
    as a dictionary.
    """
    print("* Loading training set properties.", flush=True)

    # read dictionary from CSV
    with open(csv_path, "r") as csv_file:
        reader = csv.reader(csv_file, delimiter=";")
        csv_dict = dict(reader)

    # create `properties_dict` from `csv_dict`, fix any bad filetypes
    properties_dict = {}
    for key, value in csv_dict.items():

        # first determine if key is a tuple
        key = eval(key)
        if len(key) > 1:
            tuple_key = (str(key[0]), str(key[1]))
        else:
            tuple_key = key

        # then convert the values to the correct data type
        try:
            properties_dict[tuple_key] = eval(value)
        except (SyntaxError, NameError):
            properties_dict[tuple_key] = value

        # convert any `list`s to `torch.Tensor`s (for consistency)
        if isinstance(properties_dict[tuple_key], list):
            properties_dict[tuple_key] = th.Tensor(properties_dict[tuple_key])

    return properties_dict


class HDFDataset(Dataset):
    """
    Reads and collects data from an HDF file with three datasets: "nodes", "edges", and "APDs".
    """

    def __init__(self, path: str) -> None:

        self.path = path
        hdf_file = h5py.File(self.path, "r+", swmr=True)

        # load each HDF dataset
        self.nodes = hdf_file.get("nodes")
        self.edges = hdf_file.get("edges")
        self.apds = hdf_file.get("APDs")

        # get the number of elements in the dataset
        self.n_subgraphs = self.nodes.shape[0]

    def __getitem__(self, idx: int) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        # returns specific graph elements
        nodes_i = th.from_numpy(self.nodes[idx]).type(th.float32)
        edges_i = th.from_numpy(self.edges[idx]).type(th.float32)
        apd_i = th.from_numpy(self.apds[idx]).type(th.float32)

        return (nodes_i, edges_i, apd_i)

    def __len__(self) -> int:
        # returns the number of graphs in the dataset
        return self.n_subgraphs


class BlockDataset(Dataset):
    """
    Modified `Dataset` class which returns BLOCKS of data when `__getitem__()` is called.
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        block_size: int = 10000,
    ) -> None:

        assert block_size >= batch_size

        self.block_size = block_size  # `int`
        self.batch_size = batch_size  # `int`
        self.dataset = dataset  # `HDFDataset`

    def __getitem__(self, idx: int) -> th.Tensor:
        # returns a block of data from the dataset
        start = idx * self.block_size
        end = min((idx + 1) * self.block_size, len(self.dataset))
        return self.dataset[start:end]

    def __len__(self) -> int:
        # returns the number of blocks in the dataset
        return (len(self.dataset) + self.block_size - 1) // self.block_size


class ShuffleBlockWrapper:
    """
    Extra class used to wrap a block of data, enabling data to get shuffled *within* a block.
    """

    def __init__(self, data: th.Tensor) -> None:
        self.data = data

    def __getitem__(self, idx: int) -> th.Tensor:
        return [d[idx] for d in self.data]

    def __len__(self) -> int:
        return len(self.data[0])


class BlockDataLoader(DataLoader):
    """
    Main `DataLoader` class which has been modified so as to read training data from disk in
    blocks, as opposed to a single line at a time (as is done in the original `DataLoader` class).
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int = 100,
        block_size: int = 10000,
        shuffle: bool = True,
        n_workers: int = 0,
        pin_memory: bool = True,
        *args,
        **kwargs,
    ) -> None:
        super(BlockDataLoader, self).__init__()
        # define variables to be used throughout dataloading
        self.dataset = dataset  # `HDFDataset` object
        self.batch_size = batch_size  # `int`
        self.block_size = block_size  # `int`
        self.shuffle = shuffle  # `bool`
        self.n_workers = n_workers  # `int`
        self.pin_memory = pin_memory  # `bool`
        self.block_dataset = BlockDataset(
            self.dataset, batch_size=self.batch_size, block_size=self.block_size
        )

    def __iter__(self):
        # define a regular `DataLoader` using the `BlockDataset`
        block_loader = DataLoader(
            self.block_dataset, shuffle=self.shuffle, num_workers=self.n_workers
        )

        # define a condition for determining whether to drop the last block this is done if the
        # remainder block is very small (less than a tenth the size of a normal block)
        condition = bool(
            int(self.block_dataset.__len__() / self.block_size)
            > 1 & self.block_dataset.__len__() % self.block_size
            < self.block_size / 10
        )

        # loop through and load BLOCKS of data every iteration
        for block in block_loader:
            block = [th.squeeze(b) for b in block]

            # wrap each block in a `ShuffleBlock` so that data can be shuffled within blocks
            batch_loader = DataLoader(
                dataset=ShuffleBlockWrapper(block),
                shuffle=self.shuffle,
                batch_size=self.batch_size,
                num_workers=self.n_workers,
                pin_memory=self.pin_memory,
                drop_last=condition,
            )
            for batch in batch_loader:
                yield batch

    def __len__(self) -> int:
        # returns the number of graphs in the DataLoader
        n_blocks = len(self.dataset) // self.block_size
        n_rem = len(self.dataset) % self.block_size
        n_batch_per_block = self.__ceil__(self.block_size, self.batch_size)
        n_last = self.__ceil__(n_rem, self.batch_size)
        return n_batch_per_block * n_blocks + n_last

    def __ceil__(self, x: int, y: int) -> int:
        return (x + y - 1) // y


class ProteinDataset(Dataset):
    """
    Protein data for conditioning
    """

    def __init__(
        self, protein_data_path, protein_test_idx, transform=None, *args, **kwargs
    ):
        """
        :param protein_data_path: protein data file(.smi or .csv) path
        :param transform: optional transform
        """
        # Load protein sequence data
        if protein_data_path.endswith(".smi"):
            self.protein_df = read_smi(protein_data_path, names=["Sequence"])
        elif protein_data_path.endswith(".csv"):
            self.protein_df = pd.read_csv(protein_data_path, index_col="entry_name")
        else:
            raise TypeError(
                f"{protein_data_path.split('.')[-1]} files are not supported."
            )

        self.transform = transform

        # Drop protein sequence data used in testing
        self.origin_protein_df = self.protein_df
        self.protein_df = self.protein_df.drop(self.protein_df.index[protein_test_idx])

    def __len__(self):
        return len(self.protein_df)

    def __getitem__(self, idx):
        if th.is_tensor(idx):
            idx = idx.tolist()
        sample = self.protein_df.iloc[idx].name

        if self.transform:
            sample = self.transform(sample)

        return sample
