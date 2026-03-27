from __future__ import annotations

import argparse
from pathlib import Path

from own_ai_model import (
    SimpleNeuralNetwork,
    build_datasets,
    load_config,
    load_model,
    save_history_csv,
    save_model,
    summarize_dataset,
    train_model,
)


def _validate_resumed_model_shape(model: SimpleNeuralNetwork, expected_input_size: int, expected_hidden_size: int) -> None:
    if model.input_size != expected_input_size or model.hidden_size != expected_hidden_size:
        raise ValueError(
            "The resumed model architecture does not match the current config. "
            f"Expected input={expected_input_size}, hidden={expected_hidden_size}, "
            f"received input={model.input_size}, hidden={model.hidden_size}."
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple neural network written from scratch.")
    parser.add_argument(
        "--config",
        default="config/model_config.json",
        help="Path to the JSON config file with dataset and training parameters.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce epoch-by-epoch logging.",
    )
    parser.add_argument(
        "--resume-from",
        help="Continue training from an existing saved model artifact.",
    )
    parser.add_argument(
        "--history-csv",
        help="Optional CSV export path for epoch-by-epoch training history.",
    )
    args = parser.parse_args()

    config = load_config(args.config)
    if config.model.input_size != 2:
        raise ValueError("This demo model expects exactly 2 input features.")

    train_data, test_data = build_datasets(config)
    existing_history: list[dict[str, float | int | bool]] = []
    resume_source: str | None = None
    epoch_offset = 0

    if args.resume_from:
        model, resume_payload = load_model(args.resume_from)
        _validate_resumed_model_shape(model, config.model.input_size, config.model.hidden_size)
        existing_history = [
            dict(record)
            for record in resume_payload.get("history", [])
            if isinstance(record, dict)
        ]
        training_summary = resume_payload.get("training_summary", {})
        if isinstance(training_summary, dict):
            epoch_offset = int(training_summary.get("final_epoch", len(existing_history)))
        else:
            epoch_offset = len(existing_history)
        resume_source = str(Path(args.resume_from).resolve())
    else:
        model = SimpleNeuralNetwork(
            input_size=config.model.input_size,
            hidden_size=config.model.hidden_size,
            seed=config.seed,
            hidden_activation=config.model.hidden_activation,
        )

    training_result = train_model(
        model=model,
        train_data=train_data,
        test_data=test_data,
        training_config=config.training,
        seed=config.seed + 99,
        epoch_offset=epoch_offset,
        verbose=not args.quiet,
    )

    train_summary = summarize_dataset(train_data)
    test_summary = summarize_dataset(test_data)
    combined_history = existing_history + training_result.history
    training_summary_payload = {
        **training_result.to_dict(),
        "resumed_from": resume_source,
    }
    output_path = save_model(
        path=config.artifacts.model_path,
        model=model,
        config=config,
        history=combined_history,
        final_metrics={
            "train": training_result.final_train_metrics.to_dict(),
            "test": training_result.final_test_metrics.to_dict(),
        },
        training_summary=training_summary_payload,
        dataset_summary={"train": train_summary.to_dict(), "test": test_summary.to_dict()},
    )

    history_csv_path = args.history_csv or config.artifacts.history_path
    saved_history_path: Path | None = None
    if history_csv_path:
        saved_history_path = save_history_csv(history_csv_path, combined_history)

    reloaded_model, _ = load_model(output_path)
    sample = test_data[0]
    original_probability = model.predict_probability(sample.features)
    reloaded_probability = reloaded_model.predict_probability(sample.features)
    if abs(original_probability - reloaded_probability) > 1e-12:
        raise RuntimeError("Saved model validation failed after reload.")

    print()
    print(f"Training complete. Saved parameters to: {Path(output_path).resolve()}")
    if resume_source:
        print(f"Resumed from: {resume_source}")
    print(
        f"Best epoch: {training_result.best_epoch} "
        f"after {training_result.final_epoch} total epoch(s)"
        + (" with early stopping." if training_result.stopped_early else ".")
    )
    print(f"Hidden activation: {config.model.hidden_activation}")
    print(f"Loss function: {config.training.loss_function}")
    print(f"Optimization: {config.training.optimization_method} (theta = theta - alpha * grad)")
    print(f"Final velocity norm: {training_result.final_velocity_norm:.4f}")
    print(f"Final parameter count: {training_result.final_parameter_stats.parameter_count}")
    print(f"Final parameter L2 norm: {training_result.final_parameter_stats.l2_norm:.4f}")
    print(f"Final train accuracy: {training_result.final_train_metrics.accuracy:.2%}")
    print(f"Final test accuracy:  {training_result.final_test_metrics.accuracy:.2%}")
    print(f"Final test regularization loss: {training_result.final_test_metrics.regularization_loss:.4f}")
    print(f"Final test F1 score:  {training_result.final_test_metrics.f1_score:.2%}")
    if saved_history_path is not None:
        print(f"History CSV saved to: {saved_history_path.resolve()}")
    print(
        "Sample prediction check:",
        {
            "features": sample.features,
            "label": sample.label,
            "predicted_probability": round(reloaded_probability, 4),
            "predicted_class": reloaded_model.predict_label(sample.features),
        },
    )


if __name__ == "__main__":
    main()
