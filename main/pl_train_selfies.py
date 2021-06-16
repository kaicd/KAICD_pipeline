import json
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint
from pytoda.smiles.smiles_language import SMILESLanguage

from models.pl_VAE import VAE
from datasets.pl_selfies_vae import SELFIES_VAE_lightning

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
    default="data/pretraining/language_models/selfies_language.pkl",
    help="Path to a pickle of a SMILES language object.",
)
parser.add_argument(
    "--params_filepath",
    type=str,
    default="utils/selfies.json",
    help="Path to the parameter file.",
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=200)

# Dataset args
parser = SELFIES_VAE_lightning.add_argparse_args(parser)
args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Parameter update
params = {}
with open(args.project_filepath + args.params_filepath) as f:
    params.update(json.load(f))
smiles_language = SMILESLanguage.load(
    args.project_filepath + args.smiles_language_filepath
)
params.update(
    {
        "vocab_size": smiles_language.number_of_tokens,
        "pad_index": smiles_language.padding_index,
    }
)
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

with open(args.project_filepath + args.params_filepath, "w") as f:
    json.dump(params, f)

# Define dataset and model
net = VAE(**vars(args))
data = SELFIES_VAE_lightning(device=net.device, **vars(args))

# Define pytorch-lightning Trainer multiple callbacks
on_best_loss = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_chemistry-{epoch:03d}-{val_loss:.3f}",
    monitor="val_loss",
    save_top_k=1,
    mode="min",
)
on_best_kl_div = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_chemistry-{epoch:03d}-{val_loss:.3f}",
    monitor="val_kl_div",
    save_top_k=1,
    mode="min",
)

# Define pytorch-lightning Trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    logger=loggers.WandbLogger(
        entity=args.entity, project=args.project, log_model=True
    ),
    callbacks=[on_best_loss, on_best_kl_div],
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
