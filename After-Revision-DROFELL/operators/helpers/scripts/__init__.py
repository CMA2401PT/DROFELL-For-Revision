import importlib
import argparse

# util scripts
print("target scripts")
parser = argparse.ArgumentParser()
parser.add_argument("--profile", type=str)
args, _ = parser.parse_known_args()
profile: str = args.profile

print(f"run profile: {profile}")

importlib.import_module(f"targets.scripts.{profile}")
