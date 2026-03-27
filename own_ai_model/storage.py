from __future__ import annotations

import csv
import json
from pathlib import Path

from .config import ProjectConfig
from .model import SimpleNeuralNetwork


def save_model(
    path: str | Path,
    model: SimpleNeuralNetwork,
    config: ProjectConfig,
    history: list[dict[str, float | int | bool]],
    final_metrics: dict[str, dict[str, float | int]],
    training_summary: dict[str, object] | None = None,
    dataset_summary: dict[str, dict[str, float | int]] | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "config": config.to_dict(),
        "model": model.to_dict(),
        "final_metrics": final_metrics,
        "history": history,
        "training_summary": training_summary or {},
        "dataset_summary": dataset_summary or {},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_model(path: str | Path) -> tuple[SimpleNeuralNetwork, dict[str, object]]:
    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    model = SimpleNeuralNetwork.from_dict(payload["model"])
    return model, payload


def save_history_csv(path: str | Path, history: list[dict[str, float | int | bool]]) -> Path:
    if not history:
        raise ValueError("History export requires at least one training record.")

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = list(dict.fromkeys(key for row in history for key in row.keys()))
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in history:
            writer.writerow(row)

    return output_path


def format_artifact_summary(payload: dict[str, object]) -> str:
    config = payload.get("config", {})
    model_payload = payload.get("model", {})
    training_summary = payload.get("training_summary", {})
    final_metrics = payload.get("final_metrics", {})
    dataset_summary = payload.get("dataset_summary", {})

    lines = [
        "Artifact Summary",
        f"Model input size: {model_payload.get('input_size', 'unknown')}",
        f"Model hidden size: {model_payload.get('hidden_size', 'unknown')}",
        f"Hidden activation: {model_payload.get('hidden_activation', 'unknown')}",
        f"Seed: {config.get('seed', 'unknown')}",
    ]

    training_config = config.get("training", {})
    if training_config:
        lines.extend(
            [
                f"Momentum: {training_config.get('momentum', 0.0):.2f}",
                f"Mistake focus power: {training_config.get('mistake_focus_power', 1.0):.2f}",
                f"Bias learning multiplier: {training_config.get('bias_learning_rate_multiplier', 1.0):.2f}",
                f"Loss function: {training_config.get('loss_function', 'unknown')}",
                f"Optimization: {training_config.get('optimization_method', 'unknown')}",
            ]
        )

    if training_summary:
        lines.extend(
            [
                f"Start epoch: {training_summary.get('start_epoch', 'unknown')}",
                f"Final epoch: {training_summary.get('final_epoch', 'unknown')}",
                f"Best epoch: {training_summary.get('best_epoch', 'unknown')}",
                f"Stopped early: {training_summary.get('stopped_early', False)}",
                f"Final velocity norm: {training_summary.get('final_velocity_norm', 0.0):.4f}",
            ]
        )
        parameter_stats = training_summary.get("final_parameter_stats", {})
        if parameter_stats:
            lines.extend(
                [
                    f"Parameter count: {parameter_stats.get('parameter_count', 'unknown')}",
                    f"Parameter L2 norm: {parameter_stats.get('l2_norm', 0.0):.4f}",
                ]
            )

    if final_metrics:
        test_metrics = final_metrics.get("test", {})
        lines.extend(
            [
                f"Test accuracy: {test_metrics.get('accuracy', 0.0):.2%}",
                f"Test F1 score: {test_metrics.get('f1_score', 0.0):.2%}",
                f"Test regularization loss: {test_metrics.get('regularization_loss', 0.0):.4f}",
            ]
        )

    if dataset_summary:
        train_dataset = dataset_summary.get("train", {})
        test_dataset = dataset_summary.get("test", {})
        lines.extend(
            [
                f"Train samples: {train_dataset.get('sample_count', 'unknown')}",
                f"Test samples: {test_dataset.get('sample_count', 'unknown')}",
            ]
        )

    return "\n".join(lines)
