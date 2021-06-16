import json
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytoda.proteins.protein_language import ProteinLanguage
from pytoda.smiles.smiles_language import SMILESLanguage

from models.pl_bimodal_mca import BimodalMCA_lightning
from datasets.pl_drug_affinity import DrugAffinity_lightning

parser = argparse.ArgumentParser()

# Project args
parser.add_argument("--entity", type=str, default="kaicd")
parser.add_argument("--project", type=str, default="PaccMann_sarscov2")
parser.add_argument(
    "--project_filepath",
    type=str,
    default="/home/lhs/paccmann_sarscov2/",
    help="Path to the paccmann_sarscov2 project file.",
)
parser.add_argument(
    "--save_filepath",
    type=str,
    default="/raid/PaccMann_sarscov2/paccmann_predictor/checkpoint/",
)
parser.add_argument("--seed", type=int, default=42)

# Parameter args
parser.add_argument(
    "--smiles_language_filepath",
    type=str,
    default="data/pretraining/language_models/smiles_language_chembl_gdsc_ccle_tox21_zinc_organdb_bindingdb.pkl",
    help="Path to a pickle of a SMILES language object.",
)
parser.add_argument(
    "--protein_language_filepath",
    type=str,
    default="data/pretraining/language_models/protein_language_bindingdb.pkl",
    help="Path to a pickle of a Protein language object.",
)
parser.add_argument(
    "--params_filepath",
    type=str,
    default="code/paccmann_predictor/examples/affinity/affinity.json",
    help="Path to the parameter file.",
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=200)

# Dataset args
parser = DrugAffinity_lightning.add_argparse_args(parser)
args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Parameter update
params = {}
with open(args.project_filepath + args.params_filepath) as f:
    params.update(json.load(f))
smiles_language = SMILESLanguage.load(
    args.project_filepath + args.smiles_language_filepath
)
protein_language = ProteinLanguage.load(
    args.project_filepath + args.protein_language_filepath
)
params.update(
    {
        "smiles_vocabulary_size": smiles_language.number_of_tokens,
        "protein_vocabulary_size": protein_language.number_of_tokens,
    }
)
with open(args.project_filepath + args.params_filepath, "w") as f:
    json.dump(params, f)

# Define dataset and model
net = BimodalMCA_lightning(**vars(args))
data = DrugAffinity_lightning(device=net.device, **vars(args))

# Define pytorch-lightning Trainer multiple callbacks
on_best_loss = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_predictor-{epoch:03d}-{val_loss:.3f}",
    monitor="val_loss",
    save_top_k=1,
    mode="min",
)
on_best_roc_auc = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_predictor-{epoch:03d}-{val_roc_auc:.3f}",
    monitor="val_roc_auc",
    save_top_k=1,
    mode="max",
)
on_best_avg_precision_score = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_predictor-{epoch:03d}-{val_avg_precision_score:.3f}",
    monitor="val_avg_precision_score",
    save_top_k=1,
    mode="max",
)

# Define pytorch-lightning Trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    logger=loggers.WandbLogger(
        entity=args.entity, project=args.project, log_model=True
    ),
    callbacks=[on_best_loss, on_best_roc_auc, on_best_avg_precision_score],
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
