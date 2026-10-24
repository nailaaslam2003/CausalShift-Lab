"""Command-line runner for CausalShift Lab."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from causalshift import ExperimentConfig, run_experiment


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the CausalShift Lab synthetic drift experiment.")
    parser.add_argument("--n-samples", type=int, default=3000)
    parser.add_argument("--shift-point", type=int, default=1500)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--train-fraction", type=float, default=0.6)
    parser.add_argument("--window-size", type=int, default=100)
    parser.add_argument("--confidence-threshold", type=float, default=0.70)
    parser.add_argument("--alpha", type=float, default=0.01)
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    config = ExperimentConfig(
        n_samples=args.n_samples,
        shift_point=args.shift_point,
        seed=args.seed,
        train_fraction=args.train_fraction,
        window_size=args.window_size,
        confidence_threshold=args.confidence_threshold,
        alpha=args.alpha,
    )
    result = run_experiment(config, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
