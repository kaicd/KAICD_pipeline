import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from PredictorBA_DataModule import PredictorBA_DataModule
from PredictorBA_Module import PredictorBA_Module
from PredictorDS_DataModule import PredictorDS_DataModule
from PredictorDS_Module import PredictorDS_Module

parser = argparse.ArgumentParser()
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
    default="Predictor/checkpoint/",
)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument(
    "--params_filepath",
    type=str,
    default="Config/PredictorBA.json",
    help="Path to the parameter file.",
)
parser.add_argument("--model_name", type=str, default="Predictor")

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=200)
args, _ = parser.parse_known_args()

# Model selection
if "BA" in args.params_filepath:
    Module = PredictorBA_Module
    DataModule = PredictorBA_DataModule
    monitor = ["loss", "roc_auc", "avg_precision_score"]
    mode = ["min", "max", "max"]
elif "DS" in args.params_filepath:
    Module = PredictorDS_Module
    DataModule = PredictorDS_DataModule
    monitor = ["loss", "pearson", "rmse"]
    mode = ["min", "max", "min"]
else:
    raise ValueError(
        "The params_filepath must include DS(meaning Drug Sensitivity) or BA(meaning Binding Affinity)"
    )

# Dataset args
parser = DataModule.add_argparse_args(parser)
pl.seed_everything(args.seed)
args, _ = parser.parse_known_args()

# Define dataset and model
net = Module(**vars(args))
data = DataModule(**vars(args))

# Define pytorch-lightning Trainer multiple callbacks
callbacks = []
for i, j in zip(monitor, mode):
    callbacks.append(
        ModelCheckpoint(
            dirpath=args.project_filepath + args.save_filepath,
            filename=args.model_name + "_best_" + i,
            monitor="val_" + i,
            save_top_k=1,
            mode=j,
        )
    )

# Define pytorch-lightning Trainer
trainer = pl.Trainer.from_argparse_args(
    args,
    logger=loggers.WandbLogger(
        entity=args.entity,
        project=args.project,
        name=args.model_name,
        log_model=True,
    ),
    callbacks=callbacks,
)

if args.auto_lr_find or args.auto_scale_batch_size:
    trainer.tune(net, datamodule=data)

trainer.fit(net, datamodule=data)
