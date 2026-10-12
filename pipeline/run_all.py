"""Run the whole mineral pipeline:  python -m pipeline.run_all"""
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for step in ["pipeline.s01_spectral_analysis", "pipeline.s02_train_3dcnn", "pipeline.s03_figures", "pipeline.s04_benchmark"]:
    print("\n" + "=" * 70 + f"\n{step}\n" + "=" * 70)
    if subprocess.run([sys.executable, "-m", step], cwd=ROOT).returncode != 0:
        sys.exit(f"Stopped at {step}.")
print("\nALL DONE. Results in results/ (metrics.json, model_comparison.csv, PNG figures)")
