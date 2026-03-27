from __future__ import annotations

import argparse
from pathlib import Path

from own_ai_model import benchmark_classical_models, build_datasets, load_config, save_classical_benchmark, summarize_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare classical machine-learning models on the synthetic circle dataset.")
    parser.add_argument(
        "--config",
        default="config/model_config.json",
        help="Path to the main project config.",
    )
    parser.add_argument(
        "--report-json",
        help="Optional path to save the benchmark report as JSON.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training logs.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    train_data, test_data = build_datasets(config)
    benchmark = benchmark_classical_models(
        train_data=train_data,
        test_data=test_data,
        seed=config.seed,
        logistic_epochs=config.training.epochs,
        logistic_learning_rate=min(0.18, config.training.learning_rate),
        logistic_l2_lambda=config.training.l2_lambda,
        knn_neighbors=9,
        tree_max_depth=6,
        tree_min_samples_split=max(6, config.training.batch_size),
        verbose=not args.quiet,
    )

    train_summary = summarize_dataset(train_data)
    test_summary = summarize_dataset(test_data)
    print("Classical ML Benchmark")
    print(f"Train samples: {train_summary.sample_count} | Test samples: {test_summary.sample_count}")
    print()

    for report in benchmark.reports:
        print(report.model_name)
        print(
            f"  train acc {report.train_metrics.accuracy:.2%} | "
            f"test acc {report.test_metrics.accuracy:.2%} | "
            f"test f1 {report.test_metrics.f1_score:.2%} | "
            f"test loss {report.test_metrics.loss:.4f}"
        )
        print(f"  details: {report.details}")

    print()
    print(f"Best model on test set: {benchmark.best_model_name}")

    if args.report_json:
        saved_path = save_classical_benchmark(args.report_json, benchmark)
        print(f"Saved benchmark report to: {Path(saved_path).resolve()}")


if __name__ == "__main__":
    main()
