import yaml
import subprocess
import argparse


parser = argparse.ArgumentParser()

parser.add_argument("--config_filepath", type=str, default="Config/PaccMann.yaml")
# Load yaml file (configuration file)
with open("Config/PaccMann.yaml") as f:
    cfg = yaml.load(f, Loader=yaml.FullLoader)

cmd = "python ./" + cfg["model"] + "/" + cfg["model"] + "_Main.py"

wandb_cfg = cfg["wandb"]
for key in wandb_cfg.keys():
    if wandb_cfg[key] is not None:
        cmd += " --" + key + " " + wandb_cfg[key]

model_cfg = cfg[cfg["model"] + "_filepath"]
for key in model_cfg.keys():
    if model_cfg[key] is not None:
        cmd += " --" + key + " " + model_cfg[key]

subprocess.call(cmd, shell=True)
