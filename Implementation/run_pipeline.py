from __future__ import annotations

import argparse

from src.pipeline import run_experiment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run airport gate assignment experiment.")
    parser.add_argument("--flights", type=int, default=30, help="Number of synthetic flights (20-60).")
    parser.add_argument("--gates", type=int, default=8, help="Number of gates.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--output-dir", type=str, default="outputs", help="Directory for CSV outputs.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = run_experiment(
        num_flights=args.flights,
        num_gates=args.gates,
        random_seed=args.seed,
        save_outputs=True,
        output_dir=args.output_dir,
    )
    print(f"Best model: {artifacts.model.best_model_name}")
    print("Comparison:")
    print(artifacts.comparison.to_string(index=False))
    print(f"CSV outputs written to: {args.output_dir}")


if __name__ == "__main__":
    main()

