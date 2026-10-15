"""Run the post-development V2 analyses and refresh dashboard data."""
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def main():
    stages = ("analyze_v2_extensions", "analyze_lstm_seed_sensitivity", "export_dashboard_data")
    for stage in stages:
        print(f"Running {stage}", flush=True)
        result = subprocess.run([sys.executable, "-B", str(ROOT / "scripts" / f"{stage}.py")], cwd=ROOT)
        if result.returncode:
            raise RuntimeError(f"{stage} failed with exit code {result.returncode}.")
    print("V2 analyses and dashboard export complete.")


if __name__ == "__main__":
    main()
