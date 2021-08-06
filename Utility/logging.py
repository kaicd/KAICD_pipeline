# load general packages and functions
import os
import csv
from collections import namedtuple
from typing import Union, Tuple
from warnings import filterwarnings

import torch as th
import numpy as np
import pandas as pd
import rdkit
from rdkit import RDLogger
from rdkit.Chem import Draw
import rdkit.rdBase as rkrb
import rdkit.RDLogger as rkl
from PIL import Image as pilimg
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from Utility.utils import normalize_evaluation_metrics


def disable_rdkit_logging():
    """
    Disables RDKit whiny logging.
    """
    logger = rkl.logger()
    logger.setLevel(rkl.ERROR)
    rkrb.DisableLog("rdApp.error")


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


def load_ts_properties(csv_path: str) -> dict:
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

        # convert any `list`s to `th.Tensor`s (for consistency)
        if isinstance(properties_dict[tuple_key], list):
            properties_dict[tuple_key] = th.Tensor(properties_dict[tuple_key])

    return properties_dict


def properties_to_csv(
    params: dict, prop_dict: dict, csv_filename: str, epoch_key: str, append: bool = True
) -> None:
    """
    Writes a CSV summarizing how training is going by comparing the properties of the
    generated structures during evaluation to the training set.

    Args:
    ----
        prop_dict (dict) : Contains molecular properties.
        csv_filename (str) : Full path/filename to CSV file.
        epoch_key (str) : For example, "Training set" or "Epoch {n}".
        append (bool) : Indicates whether to append to the output file (if the
          file exists) or start a new one. Default `True`.
    """
    # get all the relevant properties from the dictionary
    frac_valid = prop_dict[(epoch_key, "fraction_valid")]
    avg_n_nodes = prop_dict[(epoch_key, "avg_n_nodes")]
    avg_n_edges = prop_dict[(epoch_key, "avg_n_edges")]
    frac_unique = prop_dict[(epoch_key, "fraction_unique")]

    # use the following properties if they exist
    try:
        run_time = prop_dict[(epoch_key, "run_time")]
        frac_valid_pt = round(
            float(prop_dict[(epoch_key, "fraction_valid_properly_terminated")]), 5
        )
        frac_pt = round(
            float(prop_dict[(epoch_key, "fraction_properly_terminated")]), 5
        )
    except KeyError:
        run_time = "NA"
        frac_valid_pt = "NA"
        frac_pt = "NA"

    (
        norm_n_nodes_hist,
        norm_atom_type_hist,
        norm_formal_charge_hist,
        norm_numh_hist,
        norm_n_edges_hist,
        norm_edge_feature_hist,
        norm_chirality_hist,
    ) = normalize_evaluation_metrics(params, prop_dict, epoch_key)

    if not append:
        # file does not exist yet, create it
        with open(csv_filename, "w") as output_file:
            # write the file header
            output_file.write(
                "set, fraction_valid, fraction_valid_pt, fraction_pt, run_time, "
                "avg_n_nodes, avg_n_edges, fraction_unique, atom_type_hist, "
                "formal_charge_hist, numh_hist, chirality_hist, "
                "n_nodes_hist, n_edges_hist, edge_feature_hist\n"
            )

    # append the properties of interest to the CSV file
    with open(csv_filename, "a") as output_file:
        output_file.write(
            f"{epoch_key}, {frac_valid:.3f}, {frac_valid_pt}, {frac_pt}, {run_time}, "
            f"{avg_n_nodes:.3f}, {avg_n_edges:.3f}, {frac_unique:.3f}, "
            f"{norm_atom_type_hist}, {norm_formal_charge_hist}, "
            f"{norm_numh_hist}, {norm_chirality_hist}, {norm_n_nodes_hist}, "
            f"{norm_n_edges_hist}, {norm_edge_feature_hist}\n"
        )


def read_column(path: str, column: int) -> np.ndarray:
    """
    Reads a `column` from CSV file specified in `path` and returns it as a
    `numpy.ndarray`. Removes "NA" missing values from `data` before returning.
    """
    with open(path, "r") as csv_file:
        data = np.genfromtxt(
            csv_file,
            dtype=None,
            delimiter=",",
            skip_header=1,
            usecols=column,
            missing_values="NA",
        )
    data = np.array(data)
    data = data[~np.isnan(data)]  # exclude `nan`
    return data


def read_last_molecule_idx(restart_file_path: str) -> int:
    """
    Reads the index of the last preprocessed molecule from a file called
    "index.restart" located in the same directory as the data. Also returns the
    dataset size thus far.
    """
    with open(restart_file_path + "index.restart", "r") as txt_file:
        last_molecule_idx = np.genfromtxt(txt_file, delimiter=",")
    return int(last_molecule_idx[0]), int(last_molecule_idx[1])


def read_row(path: str, row: int, col: int) -> np.ndarray:
    """
    Reads a row from CSV file specified in `path` and returns it as a
    `numpy.ndarray`. Removes "NA" missing values from `data` before returning.
    """
    with open(path, "r") as csv_file:
        data = np.genfromtxt(
            csv_file, dtype=str, delimiter=",", skip_header=1, usecols=col
        )
    data = np.array(data)
    return data[:][row]


def suppress_warnings() -> None:
    """
    Suppresses unimportant warnings for a cleaner readout.
    """
    RDLogger.logger().setLevel(RDLogger.CRITICAL)
    filterwarnings(action="ignore", category=UserWarning)
    filterwarnings(action="ignore", category=FutureWarning)
    # could instead suppress ALL warnings with:
    # `filterwarnings(action="ignore")`
    # but choosing not to do this


def turn_off_empty_axes(n_plots_y: int, n_plots_x: int, ax: plt.axes) -> plt.axes:
    """
    Turns off empty axes in a `n_plots_y` by `n_plots_x` grid of plots.

    Args:
    ----
        n_plots_y (int) : Number of plots along the y-axis.
        n_plots_x (int) : Number of plots along the x-axis.
        ax (plt.axes) : Matplotlib object containing grid of plots.
    """
    for vi in range(n_plots_y):
        for vj in range(n_plots_x):
            # if nothing plotted on ax, it will contain `inf`
            # in axes lims, so clean up (turn off)
            if "inf" in str(ax[vi, vj].dataLim):
                ax[vi, vj].axis("off")
    return ax


def write_last_molecule_idx(
    last_molecule_idx: int, dataset_size: int, restart_file_path: str
) -> None:
    """
    Writes the index of the last preprocessed molecule (`last_molecule_idx`) and the current
    dataset size (`dataset_size`) to a file called "index.restart" to be located in the same
    directory as the data.
    """
    with open(restart_file_path + "index.restart", "w") as txt_file:
        txt_file.write(str(last_molecule_idx) + ", " + str(dataset_size))


def write_job_parameters(params: namedtuple) -> None:
    """
    Writes job parameters/hyperparameters in `params` (`namedtuple`) to CSV.
    """
    dict_path = params.job_dir + "params.csv"

    with open(dict_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        for key, value in enumerate(params._fields):
            writer.writerow([value, params[key]])


def write_preprocessing_parameters(params: namedtuple) -> None:
    """
    Writes job parameters/hyperparameters in `params` (`namedtuple`) to
    CSV, so that parameters used during preprocessing can be referenced later.
    """
    dict_path = params.dataset_dir + "preprocessing_params.csv"
    keys_to_write = [
        "atom_types",
        "formal_charge",
        "imp_H",
        "chirality",
        "group_size",
        "max_n_nodes",
        "use_aromatic_bonds",
        "use_chirality",
        "use_explicit_H",
        "ignore_H",
    ]

    with open(dict_path, "w") as csv_file:
        writer = csv.writer(csv_file, delimiter=";")
        for key, value in enumerate(params._fields):
            if value in keys_to_write:
                writer.writerow([value, params[key]])


def write_graphs_to_smi(
    smi_filename: str, molecular_graphs_list: list
) -> Tuple[float, th.Tensor]:
    """
     Writes the `TrainingGraph`s in `molecular_graphs_list` to a SMILES
    file, where the full path/filename is specified by `smi_filename`.
    """
    validity_tensor = th.zeros(len(molecular_graphs_list))

    with open(smi_filename, "w") as smi_file:

        smi_writer = rdkit.Chem.rdmolfiles.SmilesWriter(smi_file)

        for idx, molecular_graph in enumerate(molecular_graphs_list):

            mol = molecular_graph.get_molecule()
            try:
                mol.UpdatePropertyCache(strict=False)
                rdkit.Chem.SanitizeMol(mol)
                smi_writer.write(mol)
                validity_tensor[idx] = 1
            except (ValueError, RuntimeError, AttributeError):
                # molecule cannot be written to file (likely contains unphysical
                # aromatic bond(s) or an unphysical valence), so put placeholder
                smi_writer.write(
                    rdkit.Chem.MolFromSmiles("[Xe]")
                )  # `validity_tensor` remains 0

        smi_writer.close()

    fraction_valid = th.sum(validity_tensor, dim=0) / len(validity_tensor)

    return fraction_valid, validity_tensor


def write_molecules(
    save_filepath: str, molecules: list, final_nlls: th.Tensor, epoch: str
) -> Tuple[float, float]:
    """
    Writes generated molecular graphs and their NLLs. In writing the structures to a SMILES file,
    determines if structures are valid and returns this information (to avoid recomputing later).

    Args:
    ----
        molecules (list) : Contains generated `MolecularGraph`s.
        final_nlls (th.Tensor) : Contains final NLLs for the graphs.
        epoch (str) : Number corresponding to the current training epoch.

    Returns:
    -------
        fraction_valid (float) : Fraction of valid structures in the input set.
        validity_tensor (th.Tensor) : Contains either a 0 or 1 at index
          corresponding to a graph in `molecules` to indicate if graph is valid.
    """
    # save molecules as SMILES
    smi_filename = save_filepath + f"generation/epoch_{epoch}.smi"
    fraction_valid, validity_tensor = write_graphs_to_smi(
        smi_filename=smi_filename, molecular_graphs_list=molecules
    )
    # save the NLLs and validity status
    write_nlls(nll_filename=f"{smi_filename[:-3]}nll", nlls=final_nlls)
    write_validity(
        validity_file_path=f"{smi_filename[:-3]}valid", validity_tensor=validity_tensor
    )

    return fraction_valid, validity_tensor


def write_nlls(nll_filename: str, nlls: th.Tensor) -> None:
    """
    Writes the final NLL of each molecule to a file in the same order as
    the molecules are written in the corresponding SMILES file.
    """
    with open(nll_filename, "w") as nll_file:
        for nll in nlls:
            nll_file.write(f"{nll}\n")


def write_ts_properties(train_filepath: str, training_set_properties: dict) -> None:
    """
    Writes the training set properties to CSV.
    """
    training_set = train_filepath  # path to "train.smi"
    dict_path = f"{training_set[:-4]}.csv"

    with open(dict_path, "w") as csv_file:

        csv_writer = csv.writer(csv_file, delimiter=";")
        for key, value in training_set_properties.items():
            if "validity_tensor" in key:
                # skip writing the validity tensor here because it is really
                # long, instead it gets its own file elsewhere
                continue
            if isinstance(value, np.ndarray):
                csv_writer.writerow([key, list(value)])
            elif isinstance(value, th.Tensor):
                try:
                    csv_writer.writerow([key, float(value)])
                except ValueError:
                    csv_writer.writerow([key, [float(i) for i in value]])
            else:
                csv_writer.writerow([key, value])


def write_validation_scores(
    output_dir: str, epoch_key: str, model_scores: dict, append: bool = True
) -> None:
    """
    Writes a CSV with the model validation scores as a function of the epoch.

    Args:
    ----
        output_dir (str) : Full path/filename to CSV file.
        epoch_key (str) : For example, "Training set" or "Epoch {n}".
        model_scores (dict) : Contains the average NLL per molecule of {validation/train/generated}
          structures, and the average model score (weighted mean of above two scores).
        append (bool) : Indicates whether to append to the output file or start a new one.
    """
    validation_file_path = output_dir + "validation.log"

    avg_nll_val = model_scores["avg_nll_val"]
    avg_nll_train = model_scores["avg_nll_train"]
    avg_nll_gen = model_scores["avg_nll_gen"]
    uc_jsd = model_scores["UC-JSD"]

    if not append:  # create file
        with open(validation_file_path, "w") as output_file:
            # write headeres
            output_file.write(
                "set, avg_nll_per_molecule_val, avg_nll_per_molecule_train, "
                "avg_nll_per_molecule_gen, uc_jsd\n"
            )

    # append the properties of interest to the CSV file
    with open(validation_file_path, "a") as output_file:
        output_file.write(
            f"{epoch_key:}, {avg_nll_val:.5f}, {avg_nll_train:.5f}, "
            f"{avg_nll_gen:.5f}, {uc_jsd:.7f}\n"
        )


def write_validity(validity_file_path: str, validity_tensor: th.Tensor) -> None:
    """
    Writes the validity (0 or 1) of each molecule to a file in the same
    order as the molecules are written in the corresponding SMILES file.
    """
    with open(validity_file_path, "w") as valid_file:
        for valid in validity_tensor:
            valid_file.write(f"{valid}\n")
