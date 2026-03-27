from __future__ import annotations

from dataclasses import dataclass
import math
import random

from .config import TrainingConfig
from .data import Example
from .model import ModelVelocity, ParameterStats, SimpleNeuralNetwork


@dataclass(frozen=True)
class Metrics:
    loss: float
    data_loss: float
    regularization_loss: float
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "loss": self.loss,
            "data_loss": self.data_loss,
            "regularization_loss": self.regularization_loss,
            "accuracy": self.accuracy,
            "precision": self.precision,
            "recall": self.recall,
            "f1_score": self.f1_score,
            "true_positives": self.true_positives,
            "true_negatives": self.true_negatives,
            "false_positives": self.false_positives,
            "false_negatives": self.false_negatives,
        }


@dataclass(frozen=True)
class TrainingResult:
    history: list[dict[str, float | int | bool]]
    start_epoch: int
    final_epoch: int
    best_epoch: int
    epochs_ran: int
    stopped_early: bool
    final_train_metrics: Metrics
    final_test_metrics: Metrics
    final_parameter_stats: ParameterStats
    final_velocity_norm: float

    def to_dict(self) -> dict[str, object]:
        return {
            "start_epoch": self.start_epoch,
            "final_epoch": self.final_epoch,
            "best_epoch": self.best_epoch,
            "epochs_ran": self.epochs_ran,
            "stopped_early": self.stopped_early,
            "final_train_metrics": self.final_train_metrics.to_dict(),
            "final_test_metrics": self.final_test_metrics.to_dict(),
            "final_parameter_stats": self.final_parameter_stats.to_dict(),
            "final_velocity_norm": self.final_velocity_norm,
        }


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


def _learning_rate_for_epoch(initial_learning_rate: float, decay: float, local_epoch: int) -> float:
    return initial_learning_rate / (1.0 + (decay * max(0, local_epoch - 1)))


def _velocity_global_norm(velocity: ModelVelocity) -> float:
    squared_sum = velocity.b2 * velocity.b2
    for value in velocity.b1:
        squared_sum += value * value
    for value in velocity.w2:
        squared_sum += value * value
    for row in velocity.w1:
        for value in row:
            squared_sum += value * value
    return math.sqrt(squared_sum)


def iterate_batches(examples: list[Example], batch_size: int, rng: random.Random) -> list[list[Example]]:
    shuffled_examples = examples[:]
    rng.shuffle(shuffled_examples)
    return [
        shuffled_examples[start_index : start_index + batch_size]
        for start_index in range(0, len(shuffled_examples), batch_size)
    ]


def evaluate_model(
    model: SimpleNeuralNetwork,
    examples: list[Example],
    l2_lambda: float = 0.0,
    loss_function_name: str = "binary_cross_entropy",
) -> Metrics:
    if not examples:
        raise ValueError("Evaluation requires at least one example.")

    total_data_loss = 0.0
    correct_predictions = 0
    true_positives = 0
    true_negatives = 0
    false_positives = 0
    false_negatives = 0

    for example in examples:
        probability = model.predict_probability(example.features)
        prediction = 1 if probability >= 0.5 else 0

        total_data_loss += model.loss_breakdown(
            probability,
            example.label,
            loss_function_name=loss_function_name,
            l2_lambda=0.0,
        ).data_loss
        if prediction == example.label:
            correct_predictions += 1
        if prediction == 1 and example.label == 1:
            true_positives += 1
        elif prediction == 0 and example.label == 0:
            true_negatives += 1
        elif prediction == 1 and example.label == 0:
            false_positives += 1
        else:
            false_negatives += 1

    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    average_data_loss = total_data_loss / len(examples)
    regularization_loss = model.loss_breakdown(
        0.5,
        0,
        loss_function_name=loss_function_name,
        l2_lambda=l2_lambda,
    ).regularization_loss

    return Metrics(
        loss=average_data_loss + regularization_loss,
        data_loss=average_data_loss,
        regularization_loss=regularization_loss,
        accuracy=correct_predictions / len(examples),
        precision=precision,
        recall=recall,
        f1_score=f1_score,
        true_positives=true_positives,
        true_negatives=true_negatives,
        false_positives=false_positives,
        false_negatives=false_negatives,
    )


def train_model(
    model: SimpleNeuralNetwork,
    train_data: list[Example],
    test_data: list[Example],
    training_config: TrainingConfig,
    seed: int,
    epoch_offset: int = 0,
    verbose: bool = True,
) -> TrainingResult:
    if not train_data:
        raise ValueError("train_data cannot be empty.")
    if not test_data:
        raise ValueError("test_data cannot be empty.")

    rng = random.Random(seed)
    history: list[dict[str, float | int | bool]] = []

    baseline_test_metrics = evaluate_model(
        model,
        test_data,
        l2_lambda=training_config.l2_lambda,
        loss_function_name=training_config.loss_function,
    )
    best_epoch = epoch_offset
    best_test_loss = baseline_test_metrics.loss
    best_state = model.to_dict()
    epochs_without_improvement = 0
    stopped_early = False
    velocity = model.blank_velocity()

    for local_epoch in range(1, training_config.epochs + 1):
        epoch = epoch_offset + local_epoch
        current_learning_rate = _learning_rate_for_epoch(
            initial_learning_rate=training_config.learning_rate,
            decay=training_config.learning_rate_decay,
            local_epoch=local_epoch,
        )
        batches = iterate_batches(train_data, training_config.batch_size, rng)
        epoch_gradient_norm_sum = 0.0
        epoch_mistake_signal_sum = 0.0
        clipped_batches = 0
        for batch in batches:
            gradients = model.blank_gradients()
            for example in batch:
                _, mistake_signal = model.accumulate_gradients(
                    example.features,
                    example.label,
                    gradients,
                    loss_function_name=training_config.loss_function,
                    mistake_focus_power=training_config.mistake_focus_power,
                )
                epoch_mistake_signal_sum += mistake_signal
            gradient_norm, was_clipped = model.clip_gradients(gradients, training_config.gradient_clip_value)
            epoch_gradient_norm_sum += gradient_norm
            if was_clipped:
                clipped_batches += 1
            model.apply_gradients(
                gradients,
                learning_rate=current_learning_rate,
                batch_size=len(batch),
                l2_lambda=training_config.l2_lambda,
                velocity=velocity,
                momentum=training_config.momentum,
                bias_learning_rate_multiplier=training_config.bias_learning_rate_multiplier,
            )

        train_metrics = evaluate_model(
            model,
            train_data,
            l2_lambda=training_config.l2_lambda,
            loss_function_name=training_config.loss_function,
        )
        test_metrics = evaluate_model(
            model,
            test_data,
            l2_lambda=training_config.l2_lambda,
            loss_function_name=training_config.loss_function,
        )
        parameter_stats = model.parameter_stats()
        improved = test_metrics.loss < (best_test_loss - training_config.early_stopping_min_delta)
        if improved:
            best_test_loss = test_metrics.loss
            best_epoch = epoch
            best_state = model.to_dict()
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        average_gradient_norm = epoch_gradient_norm_sum / max(1, len(batches))
        average_mistake_signal = epoch_mistake_signal_sum / max(1, len(train_data))
        velocity_norm = _velocity_global_norm(velocity)
        epoch_record: dict[str, float | int | bool] = {
            "epoch": epoch,
            "learning_rate": current_learning_rate,
            "train_loss": train_metrics.loss,
            "train_data_loss": train_metrics.data_loss,
            "train_regularization_loss": train_metrics.regularization_loss,
            "train_accuracy": train_metrics.accuracy,
            "test_loss": test_metrics.loss,
            "test_data_loss": test_metrics.data_loss,
            "test_regularization_loss": test_metrics.regularization_loss,
            "test_accuracy": test_metrics.accuracy,
            "test_f1_score": test_metrics.f1_score,
            "mean_gradient_norm": average_gradient_norm,
            "mean_mistake_signal": average_mistake_signal,
            "velocity_norm": velocity_norm,
            "clipped_batches": clipped_batches,
            "parameter_l2_norm": parameter_stats.l2_norm,
            "is_best": improved,
        }
        history.append(epoch_record)

        should_report = (
            verbose
            and (
                local_epoch == 1
                or local_epoch == training_config.epochs
                or local_epoch % max(1, training_config.report_every) == 0
            )
        )
        if should_report:
            print(
                f"Epoch {epoch:>3}/{epoch_offset + training_config.epochs} | "
                f"lr {current_learning_rate:.5f} | "
                f"train loss {train_metrics.loss:.4f} | "
                f"train acc {train_metrics.accuracy:.2%} | "
                f"test acc {test_metrics.accuracy:.2%} | "
                f"test f1 {test_metrics.f1_score:.2%} | "
                f"grad {average_gradient_norm:.4f} | "
                f"mistake {average_mistake_signal:.4f}"
            )

        patience = training_config.early_stopping_patience
        if patience is not None and epochs_without_improvement >= patience:
            stopped_early = True
            if verbose:
                print(f"Early stopping triggered at epoch {epoch}. Restoring best weights from epoch {best_epoch}.")
            break

    model.load_parameters(best_state)
    final_train_metrics = evaluate_model(
        model,
        train_data,
        l2_lambda=training_config.l2_lambda,
        loss_function_name=training_config.loss_function,
    )
    final_test_metrics = evaluate_model(
        model,
        test_data,
        l2_lambda=training_config.l2_lambda,
        loss_function_name=training_config.loss_function,
    )
    final_parameter_stats = model.parameter_stats()

    return TrainingResult(
        history=history,
        start_epoch=epoch_offset,
        final_epoch=epoch_offset + len(history),
        best_epoch=best_epoch,
        epochs_ran=len(history),
        stopped_early=stopped_early,
        final_train_metrics=final_train_metrics,
        final_test_metrics=final_test_metrics,
        final_parameter_stats=final_parameter_stats,
        final_velocity_norm=_velocity_global_norm(velocity),
    )
