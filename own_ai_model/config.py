from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path


@dataclass(frozen=True)
class DatasetConfig:
    train_samples: int
    test_samples: int
    coordinate_min: float
    coordinate_max: float
    circle_radius: float
    center_x: float
    center_y: float

    def __post_init__(self) -> None:
        if self.train_samples < 2:
            raise ValueError("train_samples must be at least 2.")
        if self.test_samples < 2:
            raise ValueError("test_samples must be at least 2.")
        if self.coordinate_min >= self.coordinate_max:
            raise ValueError("coordinate_min must be smaller than coordinate_max.")
        if self.circle_radius <= 0:
            raise ValueError("circle_radius must be greater than 0.")


@dataclass(frozen=True)
class ModelConfig:
    input_size: int
    hidden_size: int
    hidden_activation: str = "relu"

    def __post_init__(self) -> None:
        if self.input_size < 1:
            raise ValueError("input_size must be at least 1.")
        if self.hidden_size < 1:
            raise ValueError("hidden_size must be at least 1.")
        if self.hidden_activation not in {"relu", "sigmoid", "tanh"}:
            raise ValueError("hidden_activation must be one of: relu, sigmoid, tanh.")


@dataclass(frozen=True)
class TrainingConfig:
    epochs: int
    learning_rate: float
    batch_size: int
    report_every: int
    loss_function: str = "binary_cross_entropy"
    optimization_method: str = "gradient_descent"
    early_stopping_patience: int | None = None
    early_stopping_min_delta: float = 0.0
    l2_lambda: float = 0.0
    gradient_clip_value: float | None = None
    learning_rate_decay: float = 0.0
    momentum: float = 0.0
    mistake_focus_power: float = 1.0
    bias_learning_rate_multiplier: float = 1.0

    def __post_init__(self) -> None:
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1.")
        if self.report_every < 1:
            raise ValueError("report_every must be at least 1.")
        if self.loss_function not in {"mean_squared_error", "binary_cross_entropy"}:
            raise ValueError("loss_function must be 'mean_squared_error' or 'binary_cross_entropy'.")
        if self.optimization_method != "gradient_descent":
            raise ValueError("optimization_method currently supports only 'gradient_descent'.")
        if self.early_stopping_patience is not None and self.early_stopping_patience < 1:
            raise ValueError("early_stopping_patience must be at least 1 when provided.")
        if self.early_stopping_min_delta < 0:
            raise ValueError("early_stopping_min_delta cannot be negative.")
        if self.l2_lambda < 0:
            raise ValueError("l2_lambda cannot be negative.")
        if self.gradient_clip_value is not None and self.gradient_clip_value <= 0:
            raise ValueError("gradient_clip_value must be greater than 0 when provided.")
        if self.learning_rate_decay < 0:
            raise ValueError("learning_rate_decay cannot be negative.")
        if not 0.0 <= self.momentum < 1.0:
            raise ValueError("momentum must be between 0.0 and less than 1.0.")
        if self.mistake_focus_power < 1.0:
            raise ValueError("mistake_focus_power must be at least 1.0.")
        if self.bias_learning_rate_multiplier <= 0:
            raise ValueError("bias_learning_rate_multiplier must be greater than 0.")


@dataclass(frozen=True)
class ArtifactConfig:
    model_path: str
    history_path: str | None = None

    def __post_init__(self) -> None:
        if not self.model_path.strip():
            raise ValueError("model_path cannot be empty.")
        if self.history_path is not None and not self.history_path.strip():
            raise ValueError("history_path cannot be empty when provided.")


@dataclass(frozen=True)
class ProjectConfig:
    seed: int
    dataset: DatasetConfig
    model: ModelConfig
    training: TrainingConfig
    artifacts: ArtifactConfig

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


def load_config(path: str | Path) -> ProjectConfig:
    config_path = Path(path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))

    return ProjectConfig(
        seed=int(raw_config["seed"]),
        dataset=DatasetConfig(**raw_config["dataset"]),
        model=ModelConfig(**raw_config["model"]),
        training=TrainingConfig(**raw_config["training"]),
        artifacts=ArtifactConfig(**raw_config["artifacts"]),
    )
