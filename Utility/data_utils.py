"""Utilities functions."""
import os
import os.path as osp
import json
import csv
import h5py
from typing import Tuple
from tqdm import tqdm
from itertools import repeat
import numpy as np
import pandas as pd
import torch as th
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch_geometric.data import Data, InMemoryDataset, download_url
from pytoda.files import read_smi
from rdkit import Chem
import networkx as nx
from six.moves import urllib

from .utils import pmap


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


class PygDataset(InMemoryDataset):
    """
        A `Pytorch Geometric <https://pytorch-geometric.readthedocs.io/en/latest/index.html>`_ data interface for datasets used in molecule generation.

        .. note::
            Some datasets may not come with any node labels, like :obj:`moses`.
            Since they don't have any properties in the original data file. The process of the
            dataset can only save the current input property and will load the same  property
            label when the processed dataset is used. You can change the augment :obj:`processed_filename`
            to re-process the dataset with intended property.

        Args:
            root (string, optional): Root directory where the dataset should be saved. (default: :obj:`./`)
            name (string, optional): The name of the dataset.  Available dataset names are as follows:
                                    :obj:`zinc250k`, :obj:`zinc_800_graphaf`, :obj:`zinc_800_jt`, :obj:`zinc250k_property`,
                                    :obj:`qm9_property`, :obj:`qm9`, :obj:`moses`. (default: :obj:`qm9`)
            prop_name (string, optional): The molecular property desired and used as the optimization target. (default: :obj:`penalized_logp`)
            conf_dict (dictionary, optional): dictionary that stores all the configuration for the corresponding dataset. Default is None, but when something is passed, it uses its information. Useful for debugging and customizing for external contributers. (default: :obj:`False`)
            transform (callable, optional): A function/transform that takes in an
                :obj:`torch_geometric.data.Data` object and returns a transformed
                version. The data object will be transformed before every access.
                (default: :obj:`None`)
            pre_transform (callable, optional): A function/transform that takes in
                an :obj:`torch_geometric.data.Data` object and returns a
                transformed version. The data object will be transformed before
                being saved to disk. (default: :obj:`None`)
            pre_filter (callable, optional): A function that takes in an
                :obj:`torch_geometric.data.Data` object and returns a boolean
                value, indicating whether the data object should be included in the
                final dataset. (default: :obj:`None`)
            use_aug (bool, optional): If :obj:`True`, data augmentation will be used. (default: :obj:`False`)
            one_shot (bool, optional): If :obj:`True`, the returned data will use one-shot format with an extra dimension of virtual node and edge feature. (default: :obj:`False`)
        """

    def __init__(self,
        project_filepath="/raid/KAICD_sarscov2/",
        preprocess_filepath="data/pretraining/ChemJTVAE/",
        filename="ZINC_500M_train",
        params_filepath="Config/ChemJTVAE.json",
        transform=None,
        pre_transform=None,
        pre_filter=None,
        use_aug=False,
        one_shot=False
    ):
        self.params = {}
        with open(params_filepath) as f:
            self.params.update(json.load(f))

        self.root = project_filepath + preprocess_filepath
        self.name = filename
        self.use_aug = use_aug
        self.one_shot = one_shot

        self.num_max_node = self.params.get("num_max_node", 25)
        self.atom_list = self.params.get("atom_list", [6, 7, 8, 9, 15, 16, 17, 35, 53])
        self.bond_type_to_int = {
            Chem.BondType.SINGLE: 0,
            Chem.BondType.DOUBLE: 1,
            Chem.BondType.TRIPLE: 2
        }

        super(PygDataset, self).__init__(
            self.root,
            transform,
            pre_transform,
            pre_filter
        )
        if osp.exists(self.processed_paths[0]):
            self.data, self.slices, self.all_smiles = th.load(self.processed_paths[0])
        else:
            self.process()

        if self.one_shot:
            self.atom_list = self.atom_list + [0]

    @property
    def raw_dir(self):
        name = 'raw'
        return osp.join(self.root, name)

    @property
    def processed_dir(self):
        name = 'processed'
        if self.one_shot:
            name = 'processed_' + 'oneshot'
        return osp.join(self.root, name)

    @property
    def raw_file_names(self):
        return self.name + ".smi"

    @property
    def processed_file_names(self):
        return self.name + ".pt"

    def download(self):
        pass

    def process(self):
        r"""Processes the dataset from raw data file to the :obj:`self.processed_dir` folder.

            If one-hot format is required, the processed data type will include an extra dimension of virtual node and edge feature.
        """

        print('Processing...')
        self.data, self.slices = self.pre_process()

        if self.pre_filter is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [data for data in data_list if self.pre_filter(data)]
            self.data, self.slices = self.collate(data_list)

        if self.pre_transform is not None:
            data_list = [self.get(idx) for idx in range(len(self))]
            data_list = [self.pre_transform(data) for data in data_list]
            self.data, self.slices = self.collate(data_list)

        print('making processed files:', self.processed_dir)
        if not osp.exists(self.processed_dir):
            os.makedirs(self.processed_dir)

        th.save((self.data, self.slices, self.all_smiles), self.processed_paths[0])
        print('Done!')

    def __repr__(self):
        return '{}({})'.format(self.name, len(self))

    def get(self, idx):
        r"""Gets the data object at index :idx:.

        Args:
            idx: The index of the data that you want to reach.
        :rtype: A data object corresponding to the input index :obj:`idx` .
        """
        data = self.data.__class__()

        if hasattr(self.data, '__num_nodes__'):
            data.num_nodes = self.data.__num_nodes__[idx]

        for key in self.data.keys:
            item, slices = self.data[key], self.slices[key]
            if th.is_tensor(item):
                s = list(repeat(slice(None), item.dim()))
                s[self.data.__cat_dim__(key, item)] = slice(slices[idx], slices[idx + 1])
            else:
                s = slice(slices[idx], slices[idx + 1])
            data[key] = item[s]

        data['smile'] = self.all_smiles[idx]

        if not self.one_shot:
            # bfs-searching order
            mol_size = data.num_atom.numpy()[0]
            pure_adj = np.sum(data.adj[:3].numpy(), axis=0)[:mol_size, :mol_size]
            if self.use_aug:
                local_perm = np.random.permutation(mol_size)
                adj_perm = pure_adj[np.ix_(local_perm, local_perm)]
                G = nx.from_numpy_matrix(np.asmatrix(adj_perm))
                start_idx = np.random.randint(adj_perm.shape[0])
            else:
                local_perm = np.arange(mol_size)
                G = nx.from_numpy_matrix(np.asmatrix(pure_adj))
                start_idx = 0

            bfs_perm = np.array(self._bfs_seq(G, start_idx))
            bfs_perm_origin = local_perm[bfs_perm]
            bfs_perm_origin = np.concatenate([bfs_perm_origin, np.arange(mol_size, self.num_max_node)])
            data.x = data.x[bfs_perm_origin]
            for i in range(4):
                data.adj[i] = data.adj[i][bfs_perm_origin][:, bfs_perm_origin]

            data['bfs_perm_origin'] = th.Tensor(bfs_perm_origin).long()

        return data

    def pre_process(self):
        input_path = self.raw_paths[0]
        input_smi = open(input_path)
        smile_list = []
        prop_list = []
        for i in input_smi:
            data = i.split(" ")
            smile_list.append(data[0])
            prop_list.append(data[1][:-1])

        self.all_smiles = smile_list
        data_list = []
        for i in tqdm(range(len(smile_list))):
            mol = Chem.MolFromSmiles(smile_list[i])
            Chem.Kekulize(mol)
            num_atom = mol.GetNumAtoms()

            if num_atom > self.num_max_node:
                continue

            if self.one_shot:
                atom_array = np.zeros((len(self.atom_list), self.num_max_node), dtype=np.int32)
                virtual_node = np.ones((1, self.num_max_node), dtype=np.int32)
            else:
                atom_array = np.zeros((self.num_max_node, len(self.atom_list)), dtype=np.float32)

            atom_idx = 0
            for atom in mol.GetAtoms():
                atom_feature = atom.GetAtomicNum()
                if self.one_shot:
                    atom_array[self.atom_list.index(atom_feature), atom_idx] = 1
                    virtual_node[0, atom_idx] = 0
                else:
                    atom_array[atom_idx, self.atom_list.index(atom_feature)] = 1
                atom_idx += 1

            if self.one_shot:
                x = th.tensor(np.concatenate((atom_array, virtual_node), axis=0))
            else:
                x = th.tensor(atom_array)

            # bonds
            adj_array = np.zeros([4, self.num_max_node, self.num_max_node], dtype=np.float32)
            for bond in mol.GetBonds():
                bond_type = bond.GetBondType()
                ch = self.bond_type_to_int[bond_type]
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()
                adj_array[ch, i, j] = 1.0
                adj_array[ch, j, i] = 1.0
            adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
            if not self.one_shot:
                adj_array += np.eye(self.num_max_node)

            data = Data(x=x)
            data.adj = th.tensor(adj_array)
            data.num_atom = num_atom
            data.y = th.tensor([float(prop_list[i])])
            data_list.append(data)

        data, slices = self.collate(data_list)

        return data, slices

    def smiles_to_graph(self, info, bond_type_to_int, atom_list, num_max_node, one_shot):
        mol = Chem.MolFromSmiles(info[0])
        Chem.Kekulize(mol)
        num_atom = mol.GetNumAtoms()

        if num_atom > num_max_node:
            return None

        if one_shot:
            atom_array = np.zeros((len(atom_list), num_max_node), dtype=np.int32)
            virtual_node = np.ones((1, num_max_node), dtype=np.int32)
        else:
            atom_array = np.zeros((num_max_node, len(atom_list)), dtype=np.float32)

        atom_idx = 0
        for atom in mol.GetAtoms():
            atom_feature = atom.GetAtomicNum()
            if one_shot:
                atom_array[atom_list.index(atom_feature), atom_idx] = 1
                virtual_node[0, atom_idx] = 0
            else:
                atom_array[atom_idx, atom_list.index(atom_feature)] = 1
            atom_idx += 1

        if one_shot:
            x = th.tensor(np.concatenate((atom_array, virtual_node), axis=0))
        else:
            x = th.tensor(atom_array)

        # bonds
        adj_array = np.zeros([4, num_max_node, num_max_node], dtype=np.float32)
        for bond in mol.GetBonds():
            bond_type = bond.GetBondType()
            ch = bond_type_to_int[bond_type]
            i = bond.GetBeginAtomIdx()
            j = bond.GetEndAtomIdx()
            adj_array[ch, i, j] = 1.0
            adj_array[ch, j, i] = 1.0
        adj_array[-1, :, :] = 1 - np.sum(adj_array, axis=0)
        if not one_shot:
            adj_array += np.eye(num_max_node)

        data = Data(x=x)
        data.adj = th.tensor(adj_array)
        data.num_atom = num_atom
        data.y = th.tensor([float(info[1])])

        return data

    def _bfs_seq(self, G, start_id):
        dictionary = dict(nx.bfs_successors(G, start_id))
        start = [start_id]
        output = [start_id]
        while len(start) > 0:
            next_vertex = []
            while len(start) > 0:
                current = start.pop(0)
                neighbor = dictionary.get(current)
                if neighbor is not None:
                    next_vertex = next_vertex + neighbor
            output = output + next_vertex
            start = next_vertex
        return output