"""Command-line entry point for the research pipeline."""

import argparse
from pathlib import Path

from src.config import load_config


def run(config_path: str | Path) -> dict:
    """Validate configuration and prepare reproducible output locations."""
    config = load_config(config_path)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MarketRisk Lab.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    config = run(args.config)
    print(f"Prepared experiment: {config['experiment']['name']}")


if __name__ == "__main__":
    main()
