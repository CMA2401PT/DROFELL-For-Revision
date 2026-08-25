import importlib
import argparse

parser=argparse.ArgumentParser()
parser.add_argument("--target",type=str,default="complex")
args,_ = parser.parse_known_args()
target=args.target

importlib.import_module(f"targets.{target}")