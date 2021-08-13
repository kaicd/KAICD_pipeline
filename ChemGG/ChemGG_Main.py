import json
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from ChemGG_Analyzer import ChemGG_Analyzer
from ChemGG_Module import ChemGG_Module
from ChemGG_DataModule import ChemGG_DataModule


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
    default="ChemGG/checkpoint/",
)
parser.add_argument("--checkpoint_filepath", type=str, default="")
parser.add_argument("--model_name", type=str, default="MNN")
parser.add_argument("--seed", type=int, default=42)

parser.add_argument(
    "--params_filepath",
    type=str,
    default="Config/ChemGG.json",
    help="Path to the parameter file.",
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp")

# Dataset args
parser = ChemGG_DataModule.add_argparse_args(parser)
args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Load parameter
params = {}
with open(args.params_filepath) as f:
    params.update(json.load(f))

# Define dataset and model
data = ChemGG_DataModule(**vars(args))
data.setup()
analyzer = ChemGG_Analyzer(
    save_filepath=args.project_filepath + "ChemGG/",
    params=params,
    train_dataloader=data.train_dataloader(),
    valid_dataloader=data.val_dataloader(),
)
net = ChemGG_Module(**vars(args), analyzer=analyzer)

# (Optional) Transfer Learning
ckpt = args.checkpoint_filepath
if not ckpt == "":
    net = ChemGG_Module.load_from_checkpoint(ckpt, **vars(args))

# Define pytorch-lightning Trainer multiple callbacks
monitor = ["loss"]
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
    max_epochs=params["CONFIG"].get("epochs", 100),
    logger=loggers.WandbLogger(
        entity=args.entity, project=args.project, name=args.model_name, log_model=True
    ),
    callbacks=callbacks,
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
