from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from .model import SimpleNeuralNetwork


@dataclass(frozen=True)
class PredictionRecord:
    row_number: int
    x: float
    y: float
    probability: float
    predicted_class: int

    def to_dict(self) -> dict[str, float | int]:
        return {
            "row_number": self.row_number,
            "x": self.x,
            "y": self.y,
            "probability": self.probability,
            "predicted_class": self.predicted_class,
        }


def _is_header_row(row: list[str]) -> bool:
    for value in row[:2]:
        try:
            float(value)
        except ValueError:
            return True
    return False


def _resolve_feature_indexes(header: list[str]) -> tuple[int, int]:
    normalized_header = [column.strip().lower() for column in header]
    x_aliases = ["x", "feature_1", "feature1", "input_1", "input1"]
    y_aliases = ["y", "feature_2", "feature2", "input_2", "input2"]

    x_index = next((normalized_header.index(alias) for alias in x_aliases if alias in normalized_header), None)
    y_index = next((normalized_header.index(alias) for alias in y_aliases if alias in normalized_header), None)

    if x_index is None or y_index is None:
        if len(header) < 2:
            raise ValueError("Prediction input files must contain at least two columns.")
        return 0, 1

    return x_index, y_index


def load_prediction_points(path: str | Path) -> list[list[float]]:
    input_path = Path(path)
    if not input_path.exists():
        raise FileNotFoundError(f"Prediction input file was not found: {input_path.resolve()}")

    with input_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.reader(handle))

    data_rows = [row for row in rows if row and any(cell.strip() for cell in row)]
    if not data_rows:
        raise ValueError("Prediction input file is empty.")

    points: list[list[float]] = []
    start_index = 0
    x_index = 0
    y_index = 1

    if _is_header_row(data_rows[0]):
        x_index, y_index = _resolve_feature_indexes(data_rows[0])
        start_index = 1

    for row_number, row in enumerate(data_rows[start_index:], start=start_index + 1):
        try:
            x_value = float(row[x_index])
            y_value = float(row[y_index])
        except (IndexError, ValueError) as error:
            raise ValueError(f"Invalid numeric values on row {row_number} in {input_path.name}.") from error
        points.append([x_value, y_value])

    return points


def predict_points(
    model: SimpleNeuralNetwork,
    points: list[list[float]],
    threshold: float = 0.5,
) -> list[PredictionRecord]:
    if not 0.0 < threshold < 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    predictions: list[PredictionRecord] = []
    for row_number, point in enumerate(points, start=1):
        probability = model.predict_probability(point)
        predictions.append(
            PredictionRecord(
                row_number=row_number,
                x=point[0],
                y=point[1],
                probability=probability,
                predicted_class=1 if probability >= threshold else 0,
            )
        )
    return predictions


def save_predictions_csv(path: str | Path, predictions: list[PredictionRecord]) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["row_number", "x", "y", "probability", "predicted_class"],
        )
        writer.writeheader()
        for prediction in predictions:
            writer.writerow(prediction.to_dict())

    return output_path
