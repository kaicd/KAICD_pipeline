"""Utilities functions."""
import re
import copy
import logging
import random
import math
from tqdm import tqdm
from math import ceil, cos, sin
from typing import Union, Tuple

import numpy as np
import pandas as pd
import torch as th
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from rdkit.Chem.rdchem import BondType
from rdkit import Chem
from rdkit.Chem import AllChem, DataStructs

import pytoda
from pytoda.transforms import Compose
from transformers import AutoTokenizer, AutoModel, pipeline

logger = logging.getLogger(__name__)


def sequential_data_preparation(
    input_batch,
    device,
    input_keep=1,
    start_index=2,
    end_index=3,
    dropout_index=1,
):
    """
    Sequential Training Data Builder.

    Args:
        input_batch (th.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
        start_index (int): The index of the sequence start token.
        end_index (int): The index of the sequence end token.
        dropout_index (int): The index of the dropout token. Defaults to 1.
        device (th.device): Device to be used.
    Returns:
    (th.Tensor, th.Tensor, th.Tensor): encoder_seq, decoder_seq,
        target_seq
        encoder_seq is a batch of padded input sequences starting with the
            start_index, of size `[sequence length +1, batch_size]`.
        decoder_seq is like encoder_seq but word dropout is applied
            (so if input_keep==1, then decoder_seq = encoder_seq).
        target_seq (th.Tensor): Batch of padded target sequences ending
            in the end_index, of size `[sequence length +1, batch_size]`.
    """
    batch_size = input_batch.shape[1]
    input_batch = input_batch.long().to(device)
    decoder_batch = input_batch.clone()
    # apply token dropout if keep != 1
    if input_keep != 1:
        # build dropout indices consisting of dropout_index
        dropout_indices = th.LongTensor(dropout_index * th.ones(1, batch_size).numpy())
        # mask for token dropout
        mask = Bernoulli(input_keep).sample((input_batch.shape[0],))
        mask = th.LongTensor(mask.numpy())
        dropout_loc = np.where(mask == 0)[0]

        decoder_batch[dropout_loc] = dropout_indices

    end_padding = th.LongTensor(th.zeros(1, batch_size).numpy())
    target_seq = th.cat((input_batch[1:, :], end_padding), dim=0)
    target_seq = copy.deepcopy(target_seq).to(device)

    return input_batch, decoder_batch, target_seq


def packed_sequential_data_preparation(
    input_batch,
    device,
    input_keep=1,
    start_index=2,
    end_index=3,
    dropout_index=1,
):
    """
    Sequential Training Data Builder.

    Args:
        input_batch (th.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
        start_index (int): The index of the sequence start token.
        end_index (int): The index of the sequence end token.
        dropout_index (int): The index of the dropout token. Defaults to 1.

    Returns:
    (th.Tensor, th.Tensor, th.Tensor): encoder_seq, decoder_seq,
        target_seq

        encoder_seq is a batch of padded input sequences starting with the
            start_index, of size `[sequence length +1, batch_size, 1]`.
        decoder_seq is like encoder_seq but word dropout is applied
            (so if input_keep==1, then decoder_seq = encoder_seq).
        target_seq (th.Tensor): Batch of padded target sequences ending
            in the end_index, of size `[sequence length +1, batch_size, 1]`.
    """

    def _process_sample(sample):
        if len(sample.shape) != 1:
            raise ValueError
        input = sample.long().to(device)
        decoder = input.clone()

        # apply token dropout if keep != 1
        if input_keep != 1:
            # mask for token dropout
            mask = Bernoulli(input_keep).sample(input.shape)
            mask = th.LongTensor(mask.numpy())
            dropout_loc = np.where(mask == 0)[0]
            decoder[dropout_loc] = dropout_index

        # just .clone() propagates to graph
        target = th.cat([input[1:].detach().clone(), th.Tensor([0]).long().to(device)])
        return input, decoder, target.to(device)

    batch = [_process_sample(sample) for sample in input_batch]

    encoder_decoder_target = zip(*batch)
    encoder_decoder_target = [
        th.nn.utils.rnn.pack_sequence(entry) for entry in encoder_decoder_target
    ]
    return encoder_decoder_target


def collate_fn(batch):
    """
    Collate function for DataLoader.

    Note: to be used as collate_fn in th.utils.data.DataLoader.

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


def packed_to_padded(seq, target_packed):
    """Converts a sequence of packed outputs into a padded tensor

    Arguments:
        seq {list} -- List of lists of length T (longest sequence) where each
            sub-list contains output tokens for relevant samples.
        E.g. [len(s) for s in seq] == [8, 8, 8, 4, 3, 1] if batch has 8 samples
        longest sequence has length 6 and only 3/8 samples have length 3.
        target_packed {list} -- Packed target sequence
    Return:
        th.Tensor: Shape bs x T (padded with 0)

    NOTE:
        Assumes that padding index is 0 and stop_index is 3
    """
    T = len(seq)
    batch_size = len(seq[0])
    padded = th.zeros(batch_size, T)

    stopped_idx = []
    target_packed += [th.Tensor()]
    # Loop over tokens per time step
    for t in range(T):
        seq_lst = seq[t].tolist()
        tg_lst = target_packed[t - 1].tolist()
        # Insert Padding token where necessary
        [seq_lst.insert(idx, 0) for idx in sorted(stopped_idx, reverse=False)]
        padded[:, t] = th.Tensor(seq_lst).long()

        stop_idx = list(filter(lambda x: tg_lst[x] == 3, range(len(tg_lst))))
        stopped_idx += stop_idx

    return padded


def unpack_sequence(seq):
    tensor_seqs, seq_lens = th.nn.utils.rnn.pad_packed_sequence(seq)
    return [s[:l] for s, l in zip(tensor_seqs.unbind(dim=1), seq_lens)]


def repack_sequence(seq):
    return th.nn.utils.rnn.pack_sequence(seq)


def perpare_packed_input(input):
    batch_sizes = input.batch_sizes
    data = []
    prev_size = 0
    for batch in batch_sizes:
        size = prev_size + batch
        data.append(input.data[prev_size:size])
        prev_size = size
    return data, batch_sizes


def manage_step_packed_vars(final_var, var, batch_size, prev_batch, batch_dim):
    if batch_size < prev_batch:
        finished_lines = prev_batch - batch_size
        break_index = var.shape[batch_dim] - finished_lines.item()
        finished_slice = slice(break_index, var.shape[batch_dim])
        # var shape: num_layers, batch, cell_size ?
        if batch_dim == 0:
            if final_var is not None:
                final_var[finished_slice, :, :] = var[finished_slice, :, :]
            var = var[:break_index, :, :]
        elif batch_dim == 1:
            if final_var is not None:
                final_var[:, finished_slice, :] = var[:, finished_slice, :]
            var = var[:, :break_index, :]
        else:
            raise ValueError("Allowed batch dim are 1 and 2")

    return final_var, var


def kl_weight(step, growth_rate=0.004):
    """Kullback-Leibler weighting function.

    KL divergence weighting for better training of
    encoder and decoder of the VAE.

    Reference:
        https://arxiv.org/abs/1511.06349

    Args:
        step (int): The training step.
        growth_rate (float): The rate at which the weight grows.
            Defaults to 0.0015 resulting in a weight of 1 around step=9000.

    Returns:
        float: The weight of KL divergence loss term.
    """
    weight = 1 / (1 + math.exp((15 - growth_rate * step)))
    return weight


def to_np(x):
    return x.data.cpu().numpy()


def crop_start_stop(smiles, smiles_language):
    """
    Arguments:
        smiles {th.Tensor} -- Shape 1 x T
    Returns:
        smiles {th.Tensor} -- Cropped away everything outside Start/Stop.
    """
    smiles = smiles.tolist()
    try:
        start_idx = smiles.index(smiles_language.start_index)
        stop_idx = smiles.index(smiles_language.stop_index)
        return smiles[start_idx + 1 : stop_idx]
    except Exception:
        return smiles


def crop_start(smiles, smiles_language):
    """
    Arguments:
        smiles {th.Tensor} -- Shape 1 x T
    Returns:
        smiles {th.Tensor} -- Cropped away everything outside Start/Stop.
    """
    smiles = smiles.tolist()
    try:
        start_idx = smiles.index(smiles_language.start_index)
        return smiles[start_idx + 1 :]
    except Exception:
        return smiles


def print_example_reconstruction(
    reconstruction, inp, language, selfies=False, crop_stop=True
):
    """[summary]

    Arguments:
        reconstruction {[type]} -- [description]
        inp {[type]} -- [description]
        language -- SMILES or ProteinLanguage object
    Raises:
        TypeError: [description]

    Returns:
        [type] -- [description]
    """
    if isinstance(language, pytoda.smiles.SMILESLanguage):
        _fn = language.token_indexes_to_smiles
        if selfies:
            _fn = Compose([_fn, language.selfies_to_smiles])
    elif isinstance(language, pytoda.proteins.ProteinLanguage):
        _fn = language.token_indexes_to_sequence
    else:
        raise TypeError(f"Unknown language class: {type(language)}")

    sample_idx = np.random.randint(len(reconstruction))

    if crop_stop:
        reconstructed = crop_start_stop(reconstruction[sample_idx], language)
    else:
        reconstructed = crop_start(reconstruction[sample_idx], language)

    # In padding mode input is tensor
    if isinstance(inp, th.Tensor):
        inp = inp.permute(1, 0)
    elif not isinstance(inp, list):
        raise TypeError(f"Unknown input type {type(inp)}")
    sample = inp[sample_idx].tolist()

    pred = _fn(reconstructed)
    target = _fn(sample)

    return target, pred


def add_avg_profile(omics_df):
    """
    To the DF of omics data, an average profile of each cancer site is added so
    as to enable a 'precision medicine regime' in which PaccMann^RL is tuned
    on the average of all profiles of a site.
    """
    # Create and append avg cell profiles
    omics_df_n = omics_df
    for site in set(omics_df["site"]):

        omics_df_n = omics_df_n.append(
            {
                "cell_line": site + "_avg",
                "cosmic_id": "avg",
                "histology": "avg",
                "site": site + "_avg",
                "gene_expression": pd.Series(
                    np.mean(
                        np.stack(
                            omics_df[
                                omics_df["site"] == site  # yapf: disable
                            ].gene_expression.values
                        ),
                        axis=0,
                    )
                ),
            },
            ignore_index=True,
        )

    return omics_df_n


def omics_data_splitter(omics_df, site, test_fraction):
    """
    Split omics data of cell lines into train and test.
    Args:
        omics_df    A pandas df of omics data for cancer cell lines
        site        The cancer site against which the generator is finetuned
        test_fraction  The fraction of cell lines in test data

    Returns:
        train_cell_lines, test_cell_lines (tuple): A tuple of lists with the
            cell line names used for training and testing
    """
    if site != "all":
        cell_lines = np.array(
            list(set(omics_df[omics_df["site"] == site]["cell_line"]))
        )
    else:
        cell_lines = np.array(list(set(omics_df["cell_line"])))
    inds = np.arange(cell_lines.shape[0])
    np.random.shuffle(inds)
    test_cell_lines = cell_lines[inds[: ceil(len(cell_lines) * test_fraction)]]
    train_cell_lines = cell_lines[inds[ceil(len(cell_lines) * test_fraction) :]]

    return train_cell_lines.tolist(), test_cell_lines.tolist()


def gaussian_mixture(batchsize, ndim, num_labels=8):
    """Generate gaussian mixture data.

    Reference: Makhzani, A., et al. "Adversarial autoencoders." (2015).

    Args:
        batchsize (int)
        ndim (int): Dimensionality of latent space/each Gaussian.
        num_labels (int, optional): Number of mixed Gaussians. Defaults to 8.

    Raises:
        Exception: ndim is not a multiple of 2.

    Returns:
        th.Tensor: samples
    """
    if ndim % 2 != 0:
        raise Exception("ndim must be a multiple of 2.")

    def sample(x, y, label, num_labels):
        shift = 1.4
        r = 2.0 * np.pi / float(num_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x_var = 0.5
    y_var = 0.05
    x = np.random.normal(0, x_var, (batchsize, ndim // 2))
    y = np.random.normal(0, y_var, (batchsize, ndim // 2))
    z = np.empty((batchsize, ndim), dtype=np.float32)
    for batch in range(batchsize):
        for zi in range(ndim // 2):
            z[batch, zi * 2 : zi * 2 + 2] = sample(
                x[batch, zi],
                y[batch, zi],
                random.randint(0, num_labels - 1),
                num_labels,
            )
    return th.Tensor(z)


def augment(x, dropout=0.0, sigma=0.0):
    """Performs augmentation on the input data batch x.

    Args:
        x (th.Tensor): Input of shape `[batch_size, input size]`.
        dropout (float, optional): Probability for each input value to be 0.
            Defaults to 0.
        sigma (float, optional): Variance of added gaussian noise to x
            (x' = x + N(0,sigma). Defaults to 0.

    Returns:
        th.Tensor: Augmented data
    """
    f = nn.Dropout(p=dropout, inplace=True)
    return f(x).add_(Normal(0, sigma).sample(x.shape).to(x.device))


def attention_list_to_matrix(coding_tuple, dim=2):
    """[summary]

    Args:
        coding_tuple (list((th.Tensor, th.Tensor))): iterable of
            (outputs, att_weights) tuples coming from the attention function
        dim (int, optional): The dimension along which expansion takes place to
            concatenate the attention weights. Defaults to 2.

    Returns:
        (th.Tensor, th.Tensor): raw_coeff, coeff

        raw_coeff: with the attention weights of all multiheads and
            convolutional kernel sizes concatenated along the given dimension,
            by default the last dimension.
        coeff: where the dimension is collapsed by averaging.
    """
    raw_coeff = th.cat([th.unsqueeze(tpl[1], 2) for tpl in coding_tuple], dim=dim)
    return raw_coeff, th.mean(raw_coeff, dim=dim)


def get_log_molar(y, ic50_max=None, ic50_min=None):
    """
    Converts PaccMann predictions from [0,1] to log(micromolar) range.
    """
    return y * (ic50_max - ic50_min) + ic50_min


def get_feature_dimensions(params: dict):
    """
    Returns dimensions of all node features.
    """
    n_atom_types = len(params["atom_types"])
    n_formal_charge = len(params["formal_charge"])
    n_numh = int(not params["use_explicit_H"] and not params["ignore_H"]) * len(
        params["imp_H"]
    )
    n_chirality = int(params["use_chirality"]) * len(params["chirality"])

    return n_atom_types, n_formal_charge, n_numh, n_chirality


def get_tensor_dimensions(
    n_atom_types: int,
    n_formal_charge: int,
    n_num_h: int,
    n_chirality: int,
    n_node_features: int,
    n_edge_features: int,
    params: dict,
):
    """
    Returns dimensions for all tensors that describe molecular graphs. Tensor dimensions are
    `list`s, except for `dim_f_term` which is  simply an `int`. Each element of the lists indicate
    the corresponding dimension of a particular subgraph matrix (i.e. `nodes`, `f_add`, etc).
    """
    max_nodes = params["max_n_nodes"]
    # define the matrix dimensions as `list`s
    # first for the graph reps...
    dim_nodes = [max_nodes, n_node_features]
    dim_edges = [max_nodes, max_nodes, n_edge_features]
    # ... then for the APDs
    if params["use_chirality"]:
        if params["use_explicit_H"] or params["ignore_H"]:
            dim_f_add = [
                params["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_chirality,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                params["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_chirality,
                n_edge_features,
            ]
    else:
        if params["use_explicit_H"] or params["ignore_H"]:
            dim_f_add = [
                params["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_edge_features,
            ]
        else:
            dim_f_add = [
                params["max_n_nodes"],
                n_atom_types,
                n_formal_charge,
                n_num_h,
                n_edge_features,
            ]
    dim_f_conn = [params["max_n_nodes"], n_edge_features]
    dim_f_term = 1

    return dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term


def get_feature_vector_indices(params: dict) -> list:
    """
    Gets the indices of the different segments of the feature vector. The indices are
    analogous to the lengths of the various segments.

    Returns:
    -------
        idc (list) : Contains the indices of the different one-hot encoded segments used in the
          feature vector representations of nodes in `MolecularGraph`s. These segments are, in
          order, atom type, formal charge, number of implicit Hs, and chirality.
    """
    idc = [params["n_atom_types"], params["n_formal_charge"]]

    # indices corresponding to implicit H's and chirality are optional (below)
    if not params["use_explicit_H"] and not params["ignore_H"]:
        idc.append(params["n_imp_H"])

    if params["use_chirality"]:
        idc.append(params["n_chirality"])

    return np.cumsum(idc).tolist()


def normalize_evaluation_metrics(
    params: dict, prop_dict: dict, epoch_key: str
) -> Tuple[th.Tensor]:
    """
    Normalizes histograms in `props_dict`, converts them to `list`s (from `th.Tensor`s)
    and rounds the elements. This is done for clarity when saving the histograms to CSV.

    Returns:
    -------
        norm_n_nodes_hist (th.Tensor) : Normalized histogram of the number of
          nodes per molecule.
        norm_atom_type_hist (th.Tensor) : Normalized histogram of the atom
          types present in the molecules.
        norm_charge_hist (th.Tensor) : Normalized histogram of the formal
          charges present in the molecules.
        norm_numh_hist (th.Tensor) : Normalized histogram of the number of
          implicit hydrogens present in the molecules.
        norm_n_edges_hist (th.Tensor) : Normalized histogram of the number of
          edges per node in the molecules.
        norm_edge_feature_hist (th.Tensor) : Normalized histogram of the
          edge features (types of bonds) present in the molecules.
        norm_chirality_hist (th.Tensor) : Normalized histogram of the
          chiral centers present in the molecules.
    """
    # compute histograms for non-optional features
    norm_n_nodes_hist = [
        round(i, 2) for i in norm(prop_dict[(epoch_key, "n_nodes_hist")]).tolist()
    ]
    norm_atom_type_hist = [
        round(i, 2) for i in norm(prop_dict[(epoch_key, "atom_type_hist")]).tolist()
    ]
    norm_charge_hist = [
        round(i, 2) for i in norm(prop_dict[(epoch_key, "formal_charge_hist")]).tolist()
    ]
    norm_n_edges_hist = [
        round(i, 2) for i in norm(prop_dict[(epoch_key, "n_edges_hist")]).tolist()
    ]
    norm_edge_feature_hist = [
        round(i, 2) for i in norm(prop_dict[(epoch_key, "edge_feature_hist")]).tolist()
    ]
    # compute histograms for optional features
    if not params["use_explicit_H"] and not params["ignore_H"]:
        norm_numh_hist = [
            round(i, 2) for i in norm(prop_dict[(epoch_key, "numh_hist")]).tolist()
        ]
    else:
        norm_numh_hist = [0] * len(params["imp_H"])
    if params["use_chirality"]:
        norm_chirality_hist = [
            round(i, 2) for i in norm(prop_dict[(epoch_key, "chirality_hist")]).tolist()
        ]
    else:
        norm_chirality_hist = [1, 0, 0]
    return (
        norm_n_nodes_hist,
        norm_atom_type_hist,
        norm_charge_hist,
        norm_numh_hist,
        norm_n_edges_hist,
        norm_edge_feature_hist,
        norm_chirality_hist,
    )


def norm(list_of_nums: list) -> list:
    """
    Normalizes input `list_of_nums` (`list` of `float`s or `int`s)
    """
    try:
        norm_list_of_nums = list_of_nums / sum(list_of_nums)
    except:  # occurs if divide by zero
        norm_list_of_nums = list_of_nums
    return norm_list_of_nums


def one_of_k_encoding(x: Union[str, int], allowable_set: list) -> "generator":
    """Returns the one-of-k encoding of a value `x` having a range of possible
    values in `allowable_set`.

    Args:
      x (str, int) : Value to be one-hot encoded.
      allowable_set (list) : `list` of all possible values.

    Returns:
      one_hot_generator (generator) : One-hot encoding. A generator of `int`s.
    """
    if x not in set(allowable_set):  # use set for speedup over list
        raise Exception(
            f"Input {x} not in allowable set {allowable_set}. Add {x} to allowable "
            f"set in either a) `features.py` or b) your submission script (`submit.py`) "
            f"and run again."
        )
    one_hot_generator = (int(x == s) for s in allowable_set)
    return one_hot_generator


def update_features(params):
    update_params = params
    # define node features
    n_atom_types, n_formal_charge, n_imp_H, n_chirality = get_feature_dimensions(
        update_params
    )
    node_features = n_atom_types + n_formal_charge + n_imp_H + n_chirality
    # define edge features
    bondtype_to_int = {BondType.SINGLE: 0, BondType.DOUBLE: 1, BondType.TRIPLE: 2}
    if update_params.get("use_aromatic_bonds", True):
        bondtype_to_int[BondType.AROMATIC] = 3
    int_to_bondtype = dict(map(reversed, bondtype_to_int.items()))
    edge_features = len(bondtype_to_int)
    # define matrix dimensions
    dim_nodes, dim_edges, dim_f_add, dim_f_conn, dim_f_term = get_tensor_dimensions(
        n_atom_types,
        n_formal_charge,
        n_imp_H,
        n_chirality,
        node_features,
        edge_features,
        update_params,
    )
    len_f_add = np.prod(dim_f_add[:])
    len_f_add_per_node = np.prod(dim_f_add[1:])
    len_f_conn = np.prod(dim_f_conn[:])
    len_f_conn_per_node = np.prod(dim_f_conn[1:])
    # update params
    feature_params = {
        "big_negative": -1e6,
        "big_positive": 1e6,
        "bondtype_to_int": bondtype_to_int,
        "int_to_bondtype": int_to_bondtype,
        "edge_features": edge_features,
        "n_atom_types": n_atom_types,
        "n_formal_charge": n_formal_charge,
        "n_imp_H": n_imp_H,
        "n_chirality": n_chirality,
        "node_features": node_features,
        "dim_nodes": dim_nodes,
        "dim_edges": dim_edges,
        "dim_f_add": dim_f_add,
        "dim_f_conn": dim_f_conn,
        "dim_f_term": dim_f_term,
        "dim_apd": [np.prod(dim_f_add) + np.prod(dim_f_conn) + 1],
        "len_f_add": len_f_add,
        "len_f_add_per_node": len_f_add_per_node,
        "len_f_conn": len_f_conn,
        "len_f_conn_per_node": len_f_conn_per_node,
    }
    update_params.update(feature_params)

    return update_params


def dgl_graph(graph, ndatakey="feat", edatakey="feat"):
    g = dgl.graph(tuple(graph["edge_index"]), num_nodes=graph["num_nodes"])
    if graph["edge_feat"] is not None:
        g.edata[edatakey] = th.from_numpy(graph["edge_feat"])
    if graph["node_feat"] is not None:
        g.ndata[ndatakey] = th.from_numpy(graph["node_feat"])
    return g


def get_fp(mol: Union[Chem.Mol, str], r=3, nBits=2048, **kwargs) -> np.ndarray:
    if isinstance(mol, str):
        mol = Chem.MolFromSmiles(mol)
    fp = AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=nBits, **kwargs)
    arr = np.zeros((0,), dtype=np.int8)
    # noinspection PyUnresolvedReferences
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def get_fingerprints(mols, r=3, nBits=2048):
    if isinstance(mols, str):
        mols = [mols]
    arrs = []
    for mol in mols:
        arrs.append(get_fp(mol, r=r, nBits=nBits))
    return np.stack(arrs)


class EmbedProt:
    def __init__(self):
        super(EmbedProt, self).__init__()
        self.tokenizer = AutoTokenizer.from_pretrained(
            "Rostlab/prot_bert_bfd", do_lower_case=False
        )
        self.model = AutoModel.from_pretrained("Rostlab/prot_bert_bfd")

    def __call__(self, proteins, device=0):
        fe = pipeline(
            "feature-extraction",
            model=self.model,
            tokenizer=self.tokenizer,
            device=device,
        )
        seqs = [" ".join(list(x)) for x in proteins]
        seqs = [re.sub(r"[UZOB]", "X", sequence) for sequence in seqs]
        embs = []
        for s in tqdm(seqs):
            emb = np.array(fe([s])[0])  # (n, 1024)
            cls = emb[0]
            rest = emb[1:]
            embs.append(np.concatenate([cls, rest]))
        return embs


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return th.squeeze(data)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, axis):
        super(Unsqueeze, self).__init__()
        self.axis = axis

    def forward(self, data):
        return th.unsqueeze(data, self.axis)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        super(Temperature, self).__init__()
        self.temperature = temperature

    def forward(self, data):
        return data / self.temperature