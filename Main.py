import yaml
import subprocess
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--model", type=str, default="Reinforce")
parser.add_argument("--mode", type=str, default="")
args, _ = parser.parse_known_args()
model = args.model
mode = args.mode if args.model == "Predictor" else ""

# Load yaml file (configuration file)
with open("Config/Main.yaml") as f:
    main_cfg = yaml.load(f, Loader=yaml.FullLoader)
with open("Config/" + model + mode + ".yaml") as f:
    model_cfg = yaml.load(f, Loader=yaml.FullLoader)

# Call the main file
cmd = "python ./" + model + "/" + model + "_Main.py"

# Command configuration(wandb)
wandb_cfg = main_cfg["wandb"]
for key in wandb_cfg.keys():
    if wandb_cfg[key] is not None:
        cmd += " --" + key + " " + wandb_cfg[key]

# Command configuration(model)
for key in model_cfg.keys():
    if model_cfg[key] is not None:
        cmd += " --" + key + " " + model_cfg[key]

subprocess.call(cmd, shell=True)
