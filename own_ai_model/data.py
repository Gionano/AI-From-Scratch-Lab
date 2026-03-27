from __future__ import annotations

from dataclasses import dataclass
import random

from .config import ProjectConfig


@dataclass(frozen=True)
class Example:
    features: list[float]
    label: int


@dataclass(frozen=True)
class DatasetSummary:
    sample_count: int
    positive_count: int
    negative_count: int
    positive_ratio: float
    x_min: float
    x_max: float
    y_min: float
    y_max: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "sample_count": self.sample_count,
            "positive_count": self.positive_count,
            "negative_count": self.negative_count,
            "positive_ratio": self.positive_ratio,
            "x_min": self.x_min,
            "x_max": self.x_max,
            "y_min": self.y_min,
            "y_max": self.y_max,
        }


def classify_point(x_value: float, y_value: float, radius: float, center_x: float, center_y: float) -> int:
    shifted_x = x_value - center_x
    shifted_y = y_value - center_y
    distance_squared = (shifted_x * shifted_x) + (shifted_y * shifted_y)
    return 1 if distance_squared <= (radius * radius) else 0


def generate_balanced_dataset(
    sample_count: int,
    coordinate_min: float,
    coordinate_max: float,
    radius: float,
    center_x: float,
    center_y: float,
    rng: random.Random,
) -> list[Example]:
    if sample_count < 2:
        raise ValueError("sample_count must be at least 2.")

    positives_needed = sample_count // 2
    negatives_needed = sample_count - positives_needed

    positives = 0
    negatives = 0
    examples: list[Example] = []
    attempts = 0
    max_attempts = sample_count * 500

    while positives < positives_needed or negatives < negatives_needed:
        x_value = rng.uniform(coordinate_min, coordinate_max)
        y_value = rng.uniform(coordinate_min, coordinate_max)
        label = classify_point(x_value, y_value, radius, center_x, center_y)
        attempts += 1

        if label == 1 and positives < positives_needed:
            examples.append(Example(features=[x_value, y_value], label=label))
            positives += 1
        elif label == 0 and negatives < negatives_needed:
            examples.append(Example(features=[x_value, y_value], label=label))
            negatives += 1

        if attempts > max_attempts:
            raise RuntimeError("Could not build a balanced dataset with the current configuration.")

    rng.shuffle(examples)
    return examples


def summarize_dataset(examples: list[Example]) -> DatasetSummary:
    if not examples:
        raise ValueError("Cannot summarize an empty dataset.")

    x_values = [example.features[0] for example in examples]
    y_values = [example.features[1] for example in examples]
    positive_count = sum(example.label for example in examples)
    negative_count = len(examples) - positive_count

    return DatasetSummary(
        sample_count=len(examples),
        positive_count=positive_count,
        negative_count=negative_count,
        positive_ratio=positive_count / len(examples),
        x_min=min(x_values),
        x_max=max(x_values),
        y_min=min(y_values),
        y_max=max(y_values),
    )


def build_datasets(config: ProjectConfig) -> tuple[list[Example], list[Example]]:
    dataset_config = config.dataset
    train_rng = random.Random(config.seed)
    test_rng = random.Random(config.seed + 1)

    train_data = generate_balanced_dataset(
        sample_count=dataset_config.train_samples,
        coordinate_min=dataset_config.coordinate_min,
        coordinate_max=dataset_config.coordinate_max,
        radius=dataset_config.circle_radius,
        center_x=dataset_config.center_x,
        center_y=dataset_config.center_y,
        rng=train_rng,
    )
    test_data = generate_balanced_dataset(
        sample_count=dataset_config.test_samples,
        coordinate_min=dataset_config.coordinate_min,
        coordinate_max=dataset_config.coordinate_max,
        radius=dataset_config.circle_radius,
        center_x=dataset_config.center_x,
        center_y=dataset_config.center_y,
        rng=test_rng,
    )
    return train_data, test_data
