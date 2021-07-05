import argparse
import subprocess

parser = argparse.ArgumentParser()

# Module select
parser.add_argument("--module", type=str, default=affinity)
