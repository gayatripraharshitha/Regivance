"""Command-line entry point for the research pipeline."""

import argparse
from pathlib import Path

from src.config import load_config
from src.data.loader import build_modeling_frame, load_industry_returns


def run(config_path: str | Path) -> dict:
    """Create the processed modeling dataset when raw data is available."""
    config = load_config(config_path)
    Path("reports/figures").mkdir(parents=True, exist_ok=True)
    Path("reports/tables").mkdir(parents=True, exist_ok=True)
    raw_path = Path(config["data"]["raw_path"])
    processed_path = Path(config["data"]["processed_path"])
    if raw_path.exists():
        processed_path.parent.mkdir(parents=True, exist_ok=True)
        returns = load_industry_returns(raw_path)
        modeling_frame = build_modeling_frame(returns, config["forecast"]["horizon_days"])
        modeling_frame.to_parquet(processed_path)
    return config


def main() -> None:
    parser = argparse.ArgumentParser(description="Run MarketRisk Lab.")
    parser.add_argument("--config", default="configs/experiment.yaml")
    args = parser.parse_args()
    config = run(args.config)
    print(f"Prepared experiment: {config['experiment']['name']}")


if __name__ == "__main__":
    main()
