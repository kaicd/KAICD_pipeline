import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from models.pl_reinforce_proteins import Reinforce_lightning
from datasets.pl_generate import Generate_lightning

parser = argparse.ArgumentParser()

# Project args
parser.add_argument("--entity", type=str, default="kaicd")
parser.add_argument("--project", type=str, default="PaccMann_rl_sarscov2")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--project_path",
    type=str,
    help="Path to PaccMann_SarsCov2 project",
    default="/home/lhs/paccmann_sarscov2/",
)
parser.add_argument(
    "--protein_data_path",
    type=str,
    help="Path to protein data for conditioning",
    default="data/training/merged_sequence_encoding/uniprot_covid-19.csv",
)
parser.add_argument(
    "--test_protein_id",
    type=int,
    help="ID of testing protein (LOOCV).",
    default=35,
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=100)

# Model args
parser = Reinforce_lightning.add_model_args(parser)

args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Define model and dataset
net = Reinforce_lightning(**vars(args))
data = Generate_lightning(**vars(args))

# Define pytorch_lightning Trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    logger=loggers.WandbLogger(
        entity=args.entity, project=args.project, log_model=True
    ),
    checkpoint_callback=False,
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)