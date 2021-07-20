import json
import argparse

import pytorch_lightning as pl
from pytorch_lightning import loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from Toxicity_Module import Toxicity_Module
from Toxicity_DataModule import Toxicity_DataModule

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
    default="paccmann_toxsmi/checkpoint/",
)
parser.add_argument("--seed", type=int, default=42)

# Parameter args
parser.add_argument(
    "--params_filepath",
    type=str,
    default="Config/Toxicity.json",
    help="Path to the parameter file.",
)
parser.add_argument(
    "--embedding_path",
    type=str,
    default="data/pretraining/toxicity_predictor/smiles_vae_embeddings.pkl",
    help="Path to the smiles embedding file.",
)

# Trainer args
parser = pl.Trainer.add_argparse_args(parser)
parser.set_defaults(gpus=1, accelerator="ddp", max_epochs=200)

# Dataset args
parser = Toxicity_DataModule.add_argparse_args(parser)
args, _ = parser.parse_known_args()
pl.seed_everything(args.seed)

# Parameter update
params = {}
with open(args.params_filepath) as f:
    params.update(json.load(f))
if params.get("embedding", "learned") == "pretrained":
    params.update({"embedding_path": args.project_filepath + args.embedding_path})
    with open(args.params_filepath, "w") as f:
        json.dump(params, f)

# Define dataset and model
net = Toxicity_Module(**vars(args))
data = Toxicity_DataModule(device=net.device, **vars(args))

# Define pytorch-lightning Trainer multiple callbacks
on_best_loss = ModelCheckpoint(
    dirpath=args.project_filepath + args.save_filepath,
    filename="paccmann_toxsmi_best_loss",
    monitor="val_loss",
    save_top_k=1,
    mode="min",
)
on_best_roc_auc = ModelCheckpoint(
    dirpath=args.project_filepath + args.save_filepath,
    filename="paccmann_toxsmi_best_roc_auc",
    monitor="val_roc_auc",
    save_top_k=1,
    mode="max",
)
on_best_avg_precision_score = ModelCheckpoint(
    dirpath=args.project_filepath + args.save_filepath,
    filename="paccmann_toxsmi_best_avg_prec",
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
