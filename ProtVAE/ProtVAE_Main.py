import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from ProtVAE_DataModule import ProtVAE_DataModule
from ProtVAE_Module import ProtVAE_Module

parser = argparse.ArgumentParser()

# Project args
parser.add_argument("--entity", type=str, default="kaicd")
parser.add_argument("--project", type=str, default="KAICD_sarscov2")
parser.add_argument(
    "--project_filepath",
    type=str,
    default="/raid/KAICD_sarscov2/",
    help="Path to the KAICD_sarscov2 project file.",
)
parser.add_argument(
    "--save_filepath",
    type=str,
    default="ProtVAE/checkpoint/",
)
parser.add_argument("--model_name", type=str, default="ProtVAE")
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--params_filepath",
    type=str,
    default="Config/ProtVAE.json",
    help="Path to the parameter file.",
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=100)

# Dataset args
parser = ProtVAE_DataModule.add_argparse_args(parser)
args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Define dataset and model
net = ProtVAE_Module(**vars(args))
data = ProtVAE_DataModule(**vars(args))

# Define pytorch-lightning Trainer multiple callbacks
monitor = ["loss", "rec", "kld"]
callbacks = []
for i in monitor:
    callbacks.append(
        ModelCheckpoint(
            dirpath=args.project_filepath + args.save_filepath,
            filename=args.model_name + "_best_" + i,
            monitor="val_" + i,
            save_top_k=1,
            mode="min",
        )
    )

# Define pytorch-lightning Trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    logger=loggers.WandbLogger(
        entity=args.entity, project=args.project, name=args.model_name, log_model=True
    ),
    callbacks=callbacks,
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
