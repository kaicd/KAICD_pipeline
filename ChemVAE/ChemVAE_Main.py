import json
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytoda.smiles.smiles_language import SMILESLanguage

from ChemVAE_Module import ChemVAE_Module
from ChemVAE_DataModule import ChemVAE_DataModule

parser = argparse.ArgumentParser()

# Project args
parser.add_argument("--entity", type=str, default="kaicd")
parser.add_argument("--project", type=str, default="PaccMann_sarscov2")
parser.add_argument(
    "--project_filepath",
    type=str,
    default="/raid/PaccMann_sarscov2/",
    help="Path to the paccmann_sarscov2 project file.",
)
parser.add_argument(
    "--save_filepath",
    type=str,
    default="/raid/PaccMann_sarscov2/paccmann_chemistry/checkpoint/",
)
parser.add_argument("--seed", type=int, default=42)

# Parameter args
parser.add_argument(
    "--smiles_language_filepath",
    type=str,
    default="Config/selfies_language.pkl",
    help="Path to a pickle of a SMILES language object.",
)
parser.add_argument(
    "--params_filepath",
    type=str,
    default="Config/ChemVAE.json",
    help="Path to the parameter file.",
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=50)

# Dataset args
parser = ChemVAE_DataModule.add_argparse_args(parser)
args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Parameter update
params = {}
with open(args.params_filepath) as f:
    params.update(json.load(f))
smiles_language = SMILESLanguage.load(args.smiles_language_filepath)

vocab_dict = smiles_language.index_to_token
params.update(
    {
        "start_index": list(vocab_dict.keys())[
            list(vocab_dict.values()).index("<START>")
        ],
        "end_index": list(vocab_dict.keys())[list(vocab_dict.values()).index("<STOP>")],
    }
)

if params.get("embedding", "learned") == "one_hot":
    params.update({"embedding_size": params["vocab_size"]})

# Parameter save
with open(args.params_filepath, "w") as f:
    json.dump(params, f)

# Define dataset and model
net = ChemVAE_Module(**vars(args))
data = ChemVAE_DataModule(device=net.device, **vars(args))

# Define pytorch-lightning Trainer multiple callbacks
on_best_loss = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_chemistry_best_loss",
    monitor="val_loss",
    save_top_k=1,
    mode="min",
)
on_best_kl_div = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_chemistry_best_kl_div",
    monitor="val_kl_div",
    save_top_k=1,
    mode="min",
)

# Define pytorch-lightning Trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    max_epochs=params.get("epochs, 100"),
    logger=loggers.WandbLogger(
        entity=args.entity, project=args.project, log_model=True
    ),
    callbacks=[on_best_loss, on_best_kl_div],
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
