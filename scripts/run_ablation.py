"""Run a configured baseline or FairSkew ablation."""
from pathlib import Path
import argparse
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from main import main

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/lightgcn_baseline.yaml")
    parser.add_argument("--epochs", type=int)
    arguments = parser.parse_args()
    sys.argv = [sys.argv[0], "--config", arguments.config]
    if arguments.epochs:
        sys.argv.extend(["--epochs", str(arguments.epochs)])
    main()
