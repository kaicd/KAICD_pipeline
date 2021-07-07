import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from ProtVAE_Module import ProtVAE
from ProtVAE_DataModule import Protein_VAE_lightning

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
    default="/raid/PaccMann_sarscov2/paccmann_omics/checkpoint/",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--params_filepath",
    type=str,
    default="Config/omics.json",
    help="Path to the parameter file.",
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=100)

# Dataset args
parser = Protein_VAE_lightning.add_argparse_args(parser)
args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Define dataset and model
net = ProtVAE(**vars(args))
data = Protein_VAE_lightning(**vars(args))

# Define pytorch-lightning Trainer multiple callbacks
on_best_loss = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_omics_best_loss",
    monitor="val_loss",
    save_top_k=1,
    mode="min",
)
on_best_rec = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_omics_best_rec",
    monitor="val_rec",
    save_top_k=1,
    mode="min",
)
on_best_kld = ModelCheckpoint(
    dirpath=args.save_filepath,
    filename="paccmann_omics_best_kld",
    monitor="val_kld",
    save_top_k=1,
    mode="min",
)

# Define pytorch-lightning Trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    logger=loggers.WandbLogger(
        entity=args.entity, project=args.project, log_model=True
    ),
    callbacks=[on_best_loss, on_best_rec, on_best_kld],
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
