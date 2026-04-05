import argparse
import os
import sys


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.join(PROJECT_ROOT, "src")
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from prescription_audit.config import build_model_registry, load_config
from prescription_audit.pipeline import run_all


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to config json")
    parser.add_argument("--repeats", type=int, default=1, help="Number of repeated runs per model")
    parser.add_argument("--force-rerun", action="store_true", help="Ignore existing progress.csv and rerun")
    args = parser.parse_args()

    config = load_config(args.config)
    model_registry = build_model_registry(config)
    leaderboard, leaderboard_path = run_all(
        config,
        model_registry,
        repeats=max(1, args.repeats),
        force_rerun=args.force_rerun,
    )
    print(leaderboard)
    print(f"leaderboard: {leaderboard_path}")


if __name__ == "__main__":
    main()
