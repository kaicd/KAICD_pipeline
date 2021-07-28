"""Utilities functions."""
import os
import copy
import logging
import random
import math
from math import ceil, cos, sin

import numpy as np
import pandas as pd
from PIL import Image as pilimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sns
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.distributions.normal import Normal
from torch.distributions.bernoulli import Bernoulli
from torch.utils.data.dataset import Dataset
from rdkit.Chem import Draw
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
import pytoda
from pytoda.transforms import Compose
from pytoda.files import read_smi


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
        input_batch (torch.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
        start_index (int): The index of the sequence start token.
        end_index (int): The index of the sequence end token.
        dropout_index (int): The index of the dropout token. Defaults to 1.
        device (torch.device): Device to be used.
    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor): encoder_seq, decoder_seq,
        target_seq
        encoder_seq is a batch of padded input sequences starting with the
            start_index, of size `[sequence length +1, batch_size]`.
        decoder_seq is like encoder_seq but word dropout is applied
            (so if input_keep==1, then decoder_seq = encoder_seq).
        target_seq (torch.Tensor): Batch of padded target sequences ending
            in the end_index, of size `[sequence length +1, batch_size]`.
    """
    batch_size = input_batch.shape[1]
    input_batch = input_batch.long().to(device)
    decoder_batch = input_batch.clone()
    # apply token dropout if keep != 1
    if input_keep != 1:
        # build dropout indices consisting of dropout_index
        dropout_indices = torch.LongTensor(
            dropout_index * torch.ones(1, batch_size).numpy()
        )
        # mask for token dropout
        mask = Bernoulli(input_keep).sample((input_batch.shape[0],))
        mask = torch.LongTensor(mask.numpy())
        dropout_loc = np.where(mask == 0)[0]

        decoder_batch[dropout_loc] = dropout_indices

    end_padding = torch.LongTensor(torch.zeros(1, batch_size).numpy())
    target_seq = torch.cat((input_batch[1:, :], end_padding), dim=0)
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
        input_batch (torch.Tensor): Batch of padded sequences, output of
            nn.utils.rnn.pad_sequence(batch) of size
            `[sequence length, batch_size, 1]`.
        input_keep (float): The probability not to drop input sequence tokens
            according to a Bernoulli distribution with p = input_keep.
            Defaults to 1.
        start_index (int): The index of the sequence start token.
        end_index (int): The index of the sequence end token.
        dropout_index (int): The index of the dropout token. Defaults to 1.

    Returns:
    (torch.Tensor, torch.Tensor, torch.Tensor): encoder_seq, decoder_seq,
        target_seq

        encoder_seq is a batch of padded input sequences starting with the
            start_index, of size `[sequence length +1, batch_size, 1]`.
        decoder_seq is like encoder_seq but word dropout is applied
            (so if input_keep==1, then decoder_seq = encoder_seq).
        target_seq (torch.Tensor): Batch of padded target sequences ending
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
            mask = torch.LongTensor(mask.numpy())
            dropout_loc = np.where(mask == 0)[0]
            decoder[dropout_loc] = dropout_index

        # just .clone() propagates to graph
        target = torch.cat(
            [input[1:].detach().clone(), torch.Tensor([0]).long().to(device)]
        )
        return input, decoder, target.to(device)

    batch = [_process_sample(sample) for sample in input_batch]

    encoder_decoder_target = zip(*batch)
    encoder_decoder_target = [
        torch.nn.utils.rnn.pack_sequence(entry) for entry in encoder_decoder_target
    ]
    return encoder_decoder_target


def collate_fn(batch):
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


def packed_to_padded(seq, target_packed):
    """Converts a sequence of packed outputs into a padded tensor

    Arguments:
        seq {list} -- List of lists of length T (longest sequence) where each
            sub-list contains output tokens for relevant samples.
        E.g. [len(s) for s in seq] == [8, 8, 8, 4, 3, 1] if batch has 8 samples
        longest sequence has length 6 and only 3/8 samples have length 3.
        target_packed {list} -- Packed target sequence
    Return:
        torch.Tensor: Shape bs x T (padded with 0)

    NOTE:
        Assumes that padding index is 0 and stop_index is 3
    """
    T = len(seq)
    batch_size = len(seq[0])
    padded = torch.zeros(batch_size, T)

    stopped_idx = []
    target_packed += [torch.Tensor()]
    # Loop over tokens per time step
    for t in range(T):
        seq_lst = seq[t].tolist()
        tg_lst = target_packed[t - 1].tolist()
        # Insert Padding token where necessary
        [seq_lst.insert(idx, 0) for idx in sorted(stopped_idx, reverse=False)]
        padded[:, t] = torch.Tensor(seq_lst).long()

        stop_idx = list(filter(lambda x: tg_lst[x] == 3, range(len(tg_lst))))
        stopped_idx += stop_idx

    return padded


def unpack_sequence(seq):
    tensor_seqs, seq_lens = torch.nn.utils.rnn.pad_packed_sequence(seq)
    return [s[:l] for s, l in zip(tensor_seqs.unbind(dim=1), seq_lens)]


def repack_sequence(seq):
    return torch.nn.utils.rnn.pack_sequence(seq)


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
        smiles {torch.Tensor} -- Shape 1 x T
    Returns:
        smiles {torch.Tensor} -- Cropped away everything outside Start/Stop.
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
        smiles {torch.Tensor} -- Shape 1 x T
    Returns:
        smiles {torch.Tensor} -- Cropped away everything outside Start/Stop.
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
    if isinstance(inp, torch.Tensor):
        inp = inp.permute(1, 0)
    elif not isinstance(inp, list):
        raise TypeError(f"Unknown input type {type(inp)}")
    sample = inp[sample_idx].tolist()

    pred = _fn(reconstructed)
    target = _fn(sample)

    return target, pred


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")


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


def plot_and_compare(
    unbiased_preds, biased_preds, site, cell_line, epoch, save_path, mode, bs
):
    biased_ratio = np.round(100 * (np.sum(biased_preds < 0) / len(biased_preds)), 1)
    unbiased_ratio = np.round(
        100 * (np.sum(unbiased_preds < 0) / len(unbiased_preds)), 1
    )
    print(f"Site: {site}, cell line: {cell_line}")
    print(f"NAIVE - {mode}: Percentage of effective compounds = {unbiased_ratio}")
    print(f"BIASED - {mode}: Percentage of effective compounds = {biased_ratio}")

    fig, ax = plt.subplots()
    sns.kdeplot(
        unbiased_preds, shade=True, color="grey", label=f"Unbiased: {unbiased_ratio}% "
    )
    sns.kdeplot(
        biased_preds, shade=True, color="red", label=f"Optimized: {biased_ratio}% "
    )
    valid = f"SMILES validity: \n {round((len(biased_preds)/bs) * 100, 1)}%"
    txt = "$\mathbf{Drug \ efficacy}$: "
    handles, labels = plt.gca().get_legend_handles_labels()
    patch = mpatches.Patch(color="none", label=txt)

    handles.insert(0, patch)  # add new patches and labels to list
    labels.insert(0, txt)

    plt.legend(handles, labels, loc="upper right")
    plt.xlabel("Predicted log(micromolar IC50)")
    plt.ylabel(f"Density of generated molecules (n={bs})")
    t1 = "PaccMann$^{\mathrm{RL}}$ "
    s = site.replace("_", " ")
    c = cell_line.replace("_", " ")
    t2 = f"generator for {s} cancer. (cell: {c})"
    plt.title(t1 + t2, size=13)
    plt.text(0.67, 0.70, valid, weight="bold", transform=plt.gca().transAxes)
    plt.text(
        0.05,
        0.8,
        "Effective compounds",
        weight="bold",
        color="grey",
        transform=plt.gca().transAxes,
    )
    ax.axvspan(-10, 0, alpha=0.5, color=[0.85, 0.85, 0.85])
    plt.xlim([-4, 8])
    plt.savefig(
        os.path.join(
            save_path,
            f"results/{mode}_{cell_line}_epoch_{epoch}_eff_{biased_ratio}.pdf",
        )
    )
    plt.clf()


def plot_and_compare_proteins(
    unbiased_preds, biased_preds, protein, epoch, save_path, mode, bs
):

    biased_ratio = np.round(100 * (np.sum(biased_preds > 0.5) / len(biased_preds)), 1)
    unbiased_ratio = np.round(
        100 * (np.sum(unbiased_preds > 0.5) / len(unbiased_preds)), 1
    )
    print(f"NAIVE - {mode}: Percentage of binding compounds = {unbiased_ratio}")
    print(f"BIASED - {mode}: Percentage of binding compounds = {biased_ratio}")

    fig, ax = plt.subplots()
    sns.distplot(
        unbiased_preds,
        kde_kws={
            "shade": True,
            "alpha": 0.5,
            "linewidth": 2,
            "clip": [0, 1],
            "kernel": "cos",
        },
        color="grey",
        label=f"Unbiased: {unbiased_ratio}% ",
        kde=True,
        rug=True,
        hist=False,
    )
    sns.distplot(
        biased_preds,
        kde_kws={
            "shade": True,
            "alpha": 0.5,
            "linewidth": 2,
            "clip": [0, 1],
            "kernel": "cos",
        },
        color="red",
        label=f"Optimized: {biased_ratio}% ",
        kde=True,
        rug=True,
        hist=False,
    )
    valid = f"SMILES validity: {round((len(biased_preds)/bs) * 100, 1)}%"
    txt = "$\mathbf{Drug \ binding}$: "
    handles, labels = plt.gca().get_legend_handles_labels()
    patch = mpatches.Patch(color="none", label=txt)

    handles.insert(0, patch)  # add new patches and labels to list
    labels.insert(0, txt)

    plt.legend(handles, labels, loc="upper left")
    plt.xlabel("Predicted binding probability")
    plt.ylabel(f"Density of generated molecules")
    t1 = "PaccMann$^{\mathrm{RL}}$ "
    # protein_name = '_'.join(protein.split('=')[1].split('-')[:-1])
    # organism = protein.split('=')[-1]
    # t2 = f'generator for: {protein_name}\n({organism})'
    protein_name = protein
    t2 = f"generator for: {protein_name}\n"
    plt.title(t1 + t2, size=10)
    plt.text(
        0.55,
        0.95,
        "Predicted as binding",
        weight="bold",
        color="grey",
        transform=plt.gca().transAxes,
    )
    ax.axvspan(0.5, 1.2, alpha=0.5, color=[0.85, 0.85, 0.85])
    plt.xlim([0.0, 1.0])
    plt.savefig(
        os.path.join(
            save_path,
            f"{mode}_{protein}_epoch_{epoch}_eff_{biased_ratio}.png",
        )
    )
    plt.clf()


def plot_loss(loss, reward, epoch, cell_line, save_path, rolling=1, site="unknown"):
    loss = pd.Series(loss).rolling(rolling).mean()
    rewards = pd.Series(reward).rolling(rolling).mean()

    plt.plot(np.arange(len(loss)), loss, color="r")
    plt.ylabel("RL-loss (log softmax)", size=12).set_color("r")
    plt.xlabel("Training steps", size=12)
    # Plot KLD on second y axis
    _ = plt.twinx()
    s = site.replace("_", " ")
    _ = cell_line.replace("_", " ")
    plt.plot(np.arange(len(rewards)), rewards, color="g")
    plt.ylabel("Achieved rewards", size=12).set_color("g")
    plt.title("PaccMann$^{\mathrm{RL}}$ generator for " + s + " cancer")
    plt.savefig(os.path.join(save_path, f"results/loss_ep_{epoch}_cell_{cell_line}"))
    plt.clf()


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
        torch.Tensor: samples
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
    return torch.Tensor(z)


def augment(x, dropout=0.0, sigma=0.0):
    """Performs augmentation on the input data batch x.

    Args:
        x (torch.Tensor): Input of shape `[batch_size, input size]`.
        dropout (float, optional): Probability for each input value to be 0.
            Defaults to 0.
        sigma (float, optional): Variance of added gaussian noise to x
            (x' = x + N(0,sigma). Defaults to 0.

    Returns:
        torch.Tensor: Augmented data
    """
    f = nn.Dropout(p=dropout, inplace=True)
    return f(x).add_(Normal(0, sigma).sample(x.shape).to(x.device))


def attention_list_to_matrix(coding_tuple, dim=2):
    """[summary]

    Args:
        coding_tuple (list((torch.Tensor, torch.Tensor))): iterable of
            (outputs, att_weights) tuples coming from the attention function
        dim (int, optional): The dimension along which expansion takes place to
            concatenate the attention weights. Defaults to 2.

    Returns:
        (torch.Tensor, torch.Tensor): raw_coeff, coeff

        raw_coeff: with the attention weights of all multiheads and
            convolutional kernel sizes concatenated along the given dimension,
            by default the last dimension.
        coeff: where the dimension is collapsed by averaging.
    """
    raw_coeff = torch.cat([torch.unsqueeze(tpl[1], 2) for tpl in coding_tuple], dim=dim)
    return raw_coeff, torch.mean(raw_coeff, dim=dim)


def get_log_molar(y, ic50_max=None, ic50_min=None):
    """
    Converts PaccMann predictions from [0,1] to log(micromolar) range.
    """
    return y * (ic50_max - ic50_min) + ic50_min


def generate_mols_img(mols, sub_img_size=(512, 512), legends=None, row=2, **kwargs):
    if legends is None:
        legends = [None] * len(mols)
    res = pilimg.new(
        "RGBA",
        (
            sub_img_size[0] * row,
            sub_img_size[1] * (len(mols) // row)
            if len(mols) % row == 0
            else sub_img_size[1] * ((len(mols) // row) + 1),
        ),
    )
    for i, mol in enumerate(mols):
        res.paste(
            Draw.MolToImage(mol, sub_img_size, legend=legends[i], **kwargs),
            ((i // row) * sub_img_size[0], (i % row) * sub_img_size[1]),
        )

    return res


class Squeeze(nn.Module):
    """Squeeze wrapper for nn.Sequential."""

    def forward(self, data):
        return torch.squeeze(data)


class Unsqueeze(nn.Module):
    """Unsqueeze wrapper for nn.Sequential."""

    def __init__(self, axis):
        super(Unsqueeze, self).__init__()
        self.axis = axis

    def forward(self, data):
        return torch.unsqueeze(data, self.axis)


class Temperature(nn.Module):
    """Temperature wrapper for nn.Sequential."""

    def __init__(self, temperature):
        super(Temperature, self).__init__()
        self.temperature = temperature

    def forward(self, data):
        return data / self.temperature


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
        if torch.is_tensor(idx):
            idx = idx.tolist()
        sample = self.protein_df.iloc[idx].name

        if self.transform:
            sample = self.transform(sample)

        return sample
