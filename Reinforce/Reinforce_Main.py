import argparse
import json
import logging
import os
import sys

import wandb
import numpy as np
import pandas as pd
from PIL import Image as pilimg
from rdkit import Chem
from rdkit.Chem import Draw
from paccmann_omics.encoders import ENCODER_FACTORY

from paccmann_chemistry.models import StackGRUEncoder, StackGRUDecoder, TeacherVAE
from paccmann_chemistry.utils import get_device
from paccmann_generator import ReinforceProtein
from paccmann_generator.plot_utils import plot_and_compare_proteins, plot_loss
from paccmann_generator.utils import disable_rdkit_logging
from paccmann_predictor.models import MODEL_FACTORY
from pytoda.files import read_smi
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage


# setup logging
logging.basicConfig(stream=sys.stdout, level=logging.DEBUG)
logger = logging.getLogger("train_paccmann_rl")
logger_m = logging.getLogger("matplotlib")
logger_m.setLevel(logging.WARNING)

# yapf: disable
parser = argparse.ArgumentParser(description='PaccMann^RL training script')
parser.add_argument(
    '--mol_model_path', type=str, help='Path to chemistry model', default="/raid/paccmann-covid/models/SELFIESVAE"
)
parser.add_argument(
    '--protein_model_path', type=str, help='Path to protein model', default="/raid/paccmann-covid/models/ProteinVAE"
)
parser.add_argument(
    '--affinity_model_path', type=str, help='Path to pretrained affinity model', default="/raid/paccmann-covid/models/affinity"
)
parser.add_argument(
    '--protein_data_path', type=str, help='Path to protein data for conditioning', default="/raid/paccmann-covid/data/training/merged_sequence_encoding/uniprot_covid-19.csv"
)
parser.add_argument(
    '--params_path', type=str, help='Model params json file directory', default="Config/conditional_generator.json"
)
parser.add_argument(
    '--model_name', type=str, help='Name for the trained model.', default="paccmann_sarscov2"
)
parser.add_argument(
    '--test_protein_id', type=int, help='ID of testing protein (LOOCV).', default=35
)
parser.add_argument(
    '--unbiased_predictions_path', type=str,
    help='Path to folder with aff. preds for 3000 mols from unbiased generator', default="/raid/paccmann-covid/data/training/unbiased_predictions"
)
parser.add_argument(
    '--tox21_path', help='Optional path to Tox21 model.', default="/raid/paccmann-covid/models/Tox21"
)
parser.add_argument(
    '--organdb_path', help='Optional path to OrganDB model.'
)
parser.add_argument(
    '--site', help='Specify a site in case of using a OrganDB model.'
)
parser.add_argument(
    '--clintox_path', help='Optional path to ClinTox model.'
)
parser.add_argument(
    '--sider_path', help='Optional path to SIDER model.'
)

args = parser.parse_args()


# yapf: enable
def main(*, parser_namespace):
    disable_rdkit_logging()

    # read the params json
    params = dict()
    with open(parser_namespace.params_path) as f:
        params.update(json.load(f))

    # get params, json args take precedence
    mol_model_path = params.get("mol_model_path", parser_namespace.mol_model_path)
    protein_model_path = params.get(
        "protein_model_path", parser_namespace.protein_model_path
    )
    affinity_model_path = params.get(
        "affinity_model_path", parser_namespace.affinity_model_path
    )
    protein_data_path = params.get(
        "protein_data_path", parser_namespace.protein_data_path
    )
    model_name = params.get("model_name", parser_namespace.model_name)  # yapf: disable
    test_id = params.get(
        "test_protein_id", parser_namespace.test_protein_id
    )  # yapf: disable
    unbiased_preds_path = params.get(
        "unbiased_predictions_path", parser_namespace.unbiased_predictions_path
    )  # yapf: disable
    model_name += "_" + str(test_id)
    logger.info(f"Model with name {model_name} starts.")

    # passing optional paths to params to possibly update_reward_fn
    optional_reward_args = [
        "tox21_path",
        "organdb_path",
        "site",
        "clintox_path",
        "sider_path",
    ]
    for arg in optional_reward_args:
        if parser_namespace.__dict__[arg]:
            # json still has presedence
            params[arg] = params.get(arg, parser_namespace.__dict__[arg])

    # Load protein sequence data
    if protein_data_path.endswith(".smi"):
        protein_df = read_smi(protein_data_path, names=["Sequence"])
    elif protein_data_path.endswith(".csv"):
        protein_df = pd.read_csv(protein_data_path, index_col="entry_name")
    else:
        raise TypeError(f"{protein_data_path.split('.')[-1]} files are not supported.")

    protein_test_name = protein_df.iloc[test_id].name
    logger.info(f"Test protein is {protein_test_name}")

    version = "(fixed)"
    wandb.init(
        entity="kaicd",
        project="PaccMann^RL_sarscov2",
        name=protein_test_name + version,
        group=protein_df.iloc[test_id]["organism_name"] + version,
    )

    # Restore SMILES Model
    with open(os.path.join(mol_model_path, "model_params.json")) as f:
        mol_params = json.load(f)

    gru_encoder = StackGRUEncoder(mol_params)
    gru_decoder = StackGRUDecoder(mol_params)
    generator = TeacherVAE(gru_encoder, gru_decoder)
    generator.load(
        os.path.join(
            mol_model_path, f"weights/best_{params.get('smiles_metric', 'rec')}.pt"
        ),
        map_location=get_device(),
    )
    # Load languages
    generator_smiles_language = SMILESLanguage.load(
        os.path.join(mol_model_path, "selfies_language.pkl")
    )
    generator._associate_language(generator_smiles_language)

    # Restore protein model
    with open(os.path.join(protein_model_path, "model_params.json")) as f:
        protein_params = json.load(f)

    # Define network
    protein_encoder = ENCODER_FACTORY["dense"](protein_params)
    protein_encoder.load(
        os.path.join(
            protein_model_path,
            f"weights/best_{params.get('omics_metric', 'both')}_encoder.pt",
        ),
        map_location=get_device(),
    )
    protein_encoder.eval()

    # Restore affinity predictor
    with open(os.path.join(affinity_model_path, "model_params.json")) as f:
        predictor_params = json.load(f)
    predictor = MODEL_FACTORY["bimodal_mca"](predictor_params)
    predictor.load(
        os.path.join(
            affinity_model_path,
            f"weights/best_{params.get('p_metric', 'ROC-AUC')}_bimodal_mca.pt",
        ),
        map_location=get_device(),
    )
    predictor.eval()

    # Load languages
    affinity_smiles_language = SMILESLanguage.load(
        os.path.join(affinity_model_path, "smiles_language.pkl")
    )
    affinity_protein_language = ProteinLanguage.load(
        os.path.join(affinity_model_path, "protein_language.pkl")
    )
    predictor._associate_language(affinity_smiles_language)
    predictor._associate_language(affinity_protein_language)

    # Specifies the baseline model used for comparison
    unbiased_preds = np.array(
        pd.read_csv(os.path.join(unbiased_preds_path, protein_test_name + ".csv"))[
            "affinity"
        ].values
    )  # yapf: disable

    # Create a fresh model that will be optimized
    gru_encoder_rl = StackGRUEncoder(mol_params)
    gru_decoder_rl = StackGRUDecoder(mol_params)
    generator_rl = TeacherVAE(gru_encoder_rl, gru_decoder_rl)
    generator_rl.load(
        os.path.join(mol_model_path, f"weights/best_{params.get('metric', 'rec')}.pt"),
        map_location=get_device(),
    )
    generator_rl.eval()
    # Load languages
    generator_rl._associate_language(generator_smiles_language)

    protein_encoder_rl = ENCODER_FACTORY["dense"](protein_params)
    protein_encoder_rl.load(
        os.path.join(
            protein_model_path,
            f"weights/best_{params.get('metric', 'both')}_encoder.pt",
        ),
        map_location=get_device(),
    )
    protein_encoder_rl.eval()
    model_folder_name = model_name
    learner = ReinforceProtein(
        generator_rl,
        protein_encoder_rl,
        predictor,
        protein_df,
        params,
        model_folder_name,
        logger,
    )

    global_step = 0
    biased_ratios, tox_ratios = [], []
    rewards, rl_losses = [], []
    gen_mols, gen_prot, gen_affinity, mode, toxes = [], [], [], [], []
    non_toxic_useful_smiles, non_toxic_useful_preds = [], []

    logger.info(f"Model stored at {learner.model_path}")

    train_protein_df = protein_df.drop(protein_df.index[test_id])
    for epoch in range(1, params["epochs"] + 1):

        for step in range(1, len(train_protein_df) + 1):
            # Randomly sample a protein
            protein_name = train_protein_df.iloc[step - 1].name
            logger.info(f"Current train protein: {protein_name}")

            rew, loss = learner.policy_gradient(
                protein_name, epoch, params["batch_size"]
            )
            rew = rew.item()
            loss = loss.item()
            logger.info(
                f"Epoch {epoch:d}/{params['epochs']:d}, step {step:d}/"
                f"{len(train_protein_df):d}\t loss={loss:.2f}, mean rew={rew:.2f}"
            )

            wandb.log({"rl_loss": loss, "global_step": global_step})
            wandb.log({"rewards": rew, "global_step": global_step})
            rewards.append(rew)
            rl_losses.append(loss)
            global_step += 1

        logger.info(f"EVAL protein: {protein_test_name}")

        smiles, preds = learner.generate_compounds_and_evaluate(
            epoch, params["eval_batch_size"], protein_test_name
        )
        preds = preds.cpu().numpy()
        gs = [s for i, s in enumerate(smiles) if preds[i] > 0.5]
        gp = preds[preds > 0.5]
        for p, s in zip(gp, gs):
            gen_mols.append(s)
            gen_prot.append(protein_test_name)
            gen_affinity.append(p)
            mode.append("eval")

            tox = learner.tox21(s)
            toxes.append(tox)
            if tox == 1.0:
                non_toxic_useful_smiles.append(s)
                non_toxic_useful_preds.append(p)

        plot_and_compare_proteins(
            unbiased_preds,
            preds,
            protein_test_name,
            epoch,
            learner.model_path,
            "train",
            params["eval_batch_size"],
        )
        biased_ratio = np.round(100 * (np.sum(preds > 0.5) / len(preds)), 1)
        biased_ratios.append(biased_ratio)

        all_toxes = np.array([learner.tox21(s) for s in smiles])
        tox_ratio = np.round(100 * (np.sum(all_toxes == 1.0) / len(all_toxes)), 1)
        tox_ratios.append(tox_ratio)

        logger.info(f"Percentage of non-toxic compounds {tox_ratios[-1]}")

        wandb.log({"efficacy_ratio": biased_ratio, "epoch": epoch})
        wandb.log({"tox_ratio": tox_ratio, "epoch": epoch})
        img = os.path.join(
            learner.model_path,
            f"results/train_{protein_test_name}_epoch_{epoch}_eff_{biased_ratio}.png",
        )
        wandb.log(
            {"NAIVE and BIASED binding compounds distribution": [wandb.Image(img)]}
        )

        # Log top 4 generate molecule
        idx = np.argsort(non_toxic_useful_preds)[::-1]
        lead = []
        captions = []
        for i in idx:
            mol = Chem.MolFromSmiles(non_toxic_useful_smiles[i])
            if mol:
                lead.append(mol)
                captions.append(str(non_toxic_useful_preds[i]))
                if len(lead) == 4:
                    break

        if len(lead) > 0:
            wandb.log(
                {
                    "Top N Generative Molecules": [
                        wandb.Image(generate_mols_img(lead, legends=captions))
                    ]
                }
            )

    df = pd.DataFrame(
        {
            "protein": gen_prot,
            "SMILES": gen_mols,
            "Binding probability": gen_affinity,
            "mode": mode,
            "Tox21": toxes,
        }
    )
    df.to_csv(os.path.join(learner.model_path, "results", "generated.csv"))
    wandb.log({"Results": wandb.Table(dataframe=df)})

    learner.save(f"gen_{params['epochs']}.pt", f"enc_{params['epochs']}.pt")


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


if __name__ == "__main__":
    main(parser_namespace=args)
