import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from Reinforce.Reinforce_Module import Reinforce
from Reinforce.Reinforce_DataModule import Reinforce_DataModule


parser = argparse.ArgumentParser()

# Project args
parser.add_argument("--entity", type=str, default="kaicd")
parser.add_argument("--project", type=str, default="PaccMann_rl_sarscov2")
parser.add_argument(
    "--project_path",
    type=str,
    help="Path to PaccMann_SarsCov2 project",
    default="/raid/PaccMann_sarscov2/",
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
    default="/raid/PaccMann_sarscov2/paccmann_generator/checkpoint/",
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
parser = Reinforce.add_model_args(parser)
args, _ = parser.parse_known_args()

# Define model and dataset
net = Reinforce(**vars(args))
data = Reinforce_DataModule(**vars(args))

# Define pytorch-lightning Trainer single callbacks
on_best_roc_auc = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_generator_best_rl_loss",
    monitor="rl_loss",
    save_top_k=1,
    mode="min",
)

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
