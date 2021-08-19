import argparse
import json

import pandas as pd
import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from Reinforce.Reinforce_DataModule import Reinforce_DataModule
from Reinforce.Reinforce_Module import Reinforce_Module

parser = argparse.ArgumentParser()

# Project args
parser.add_argument("--entity", type=str, default="kaicd")
parser.add_argument("--project", type=str, default="KAICD_rl_sarscov2")
parser.add_argument(
    "--project_path",
    type=str,
    help="Path to KAICD_sarscov2 project",
    default="/raid/KAICD_sarscov2/",
)
parser.add_argument(
    "--protein_data_path",
    type=str,
    help="Path to protein data for conditioning",
    default="data/training/merged_sequence_encoding/uniprot_covid-19.csv",
)
parser.add_argument(
    "--save_filepath",
    type=str,
    default="Reinforce/checkpoint/",
)
parser.add_argument(
    "--test_protein_id",
    type=int,
    help="ID of testing protein (LOOCV).",
    default=35,
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp")

# Model args
parser = Reinforce_Module.add_model_args(parser)
args, _ = parser.parse_known_args()

# Load Parameter
params = {}
with open(args.params_path) as f:
    params.update(json.load(f))

# Define model and dataset
net = Reinforce_Module(**vars(args))
data = Reinforce_DataModule(**vars(args))

# Define pytorch_lightning Trainer
wandb_group = "OTHER"
group_list = ["HUMAN", "SARS2", "CVHSA", "OTHER"]
protein_df = pd.read_csv(
    args.project_path + args.protein_data_path, index_col="entry_name"
)
wandb_name = protein_df.index[args.test_protein_id]
for group in group_list:
    if group in wandb_name:
        wandb_group = group

# Define pytorch-lightning Trainer single callbacks
on_best_rl = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename=args.model_name + "_" + wandb_name + "_best_rl_loss",
    monitor="rl_loss",
    save_top_k=1,
    mode="min",
)

trainer = pl.Trainer.from_argparse_args(
    args,
    max_epochs=params.get("epochs", 100),
    logger=loggers.WandbLogger(
        entity=args.entity,
        project=args.project,
        name=wandb_name,
        group=wandb_group,
        log_model=True,
    ),
    checkpoint_callback=[on_best_rl],
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
