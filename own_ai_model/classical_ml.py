from __future__ import annotations

from dataclasses import dataclass
import json
import math
import random
from pathlib import Path
from typing import Protocol

from .data import Example
from .trainer import Metrics


def _safe_probability(probability: float) -> float:
    return min(max(probability, 1e-7), 1.0 - 1e-7)


def _binary_cross_entropy(probability: float, label: int) -> float:
    safe_probability = _safe_probability(probability)
    return -((label * math.log(safe_probability)) + ((1 - label) * math.log(1.0 - safe_probability)))


def _safe_divide(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator else 0.0


class ProbabilisticBinaryClassifier(Protocol):
    def predict_probability(self, features: list[float]) -> float:
        ...


@dataclass(frozen=True)
class LogisticRegressionTrainingResult:
    history: list[dict[str, float | int]]
    final_metrics: Metrics

    def to_dict(self) -> dict[str, object]:
        return {
            "history": self.history,
            "final_metrics": self.final_metrics.to_dict(),
        }


@dataclass(frozen=True)
class ClassicalModelReport:
    model_name: str
    train_metrics: Metrics
    test_metrics: Metrics
    details: dict[str, float | int | str | bool]

    def to_dict(self) -> dict[str, object]:
        return {
            "model_name": self.model_name,
            "train_metrics": self.train_metrics.to_dict(),
            "test_metrics": self.test_metrics.to_dict(),
            "details": self.details,
        }


@dataclass(frozen=True)
class ClassicalBenchmark:
    reports: list[ClassicalModelReport]
    best_model_name: str

    def to_dict(self) -> dict[str, object]:
        return {
            "reports": [report.to_dict() for report in self.reports],
            "best_model_name": self.best_model_name,
        }


@dataclass
class DecisionTreeNode:
    probability: float
    sample_count: int
    depth: int
    feature_index: int | None = None
    threshold: float | None = None
    left: DecisionTreeNode | None = None
    right: DecisionTreeNode | None = None

    @property
    def is_leaf(self) -> bool:
        return self.feature_index is None


class LogisticRegressionClassifier:
    def __init__(self, input_size: int, seed: int = 0, feature_mode: str = "quadratic") -> None:
        if input_size < 1:
            raise ValueError("input_size must be at least 1.")
        if feature_mode not in {"raw", "quadratic"}:
            raise ValueError("feature_mode must be 'raw' or 'quadratic'.")

        self.input_size = input_size
        self.feature_mode = feature_mode
        transformed_size = self.transformed_input_size
        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(max(1, transformed_size))
        self.weights = [rng.uniform(-scale, scale) for _ in range(transformed_size)]
        self.bias = 0.0

    @property
    def transformed_input_size(self) -> int:
        if self.feature_mode == "raw":
            return self.input_size
        if self.input_size != 2:
            raise ValueError("quadratic feature mode currently expects exactly 2 input features.")
        return 6

    def transform_features(self, features: list[float]) -> list[float]:
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, received {len(features)}.")

        if self.feature_mode == "raw":
            return features[:]

        x_value, y_value = features
        return [
            x_value,
            y_value,
            x_value * x_value,
            y_value * y_value,
            x_value * y_value,
            (x_value * x_value) + (y_value * y_value),
        ]

    @staticmethod
    def _sigmoid(value: float) -> float:
        bounded = max(min(value, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-bounded))

    def decision_function(self, features: list[float]) -> float:
        transformed = self.transform_features(features)
        return self.bias + sum(weight * value for weight, value in zip(self.weights, transformed, strict=True))

    def predict_probability(self, features: list[float]) -> float:
        return self._sigmoid(self.decision_function(features))

    def predict_label(self, features: list[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_probability(features) >= threshold else 0

    def weight_l2_penalty(self) -> float:
        return sum(weight * weight for weight in self.weights)

    def fit(
        self,
        examples: list[Example],
        epochs: int,
        learning_rate: float,
        l2_lambda: float = 0.0,
        report_every: int = 20,
        verbose: bool = True,
    ) -> LogisticRegressionTrainingResult:
        if not examples:
            raise ValueError("examples cannot be empty.")
        if epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if learning_rate <= 0.0:
            raise ValueError("learning_rate must be greater than 0.")
        if l2_lambda < 0.0:
            raise ValueError("l2_lambda cannot be negative.")

        history: list[dict[str, float | int]] = []
        sample_count = len(examples)

        for epoch in range(1, epochs + 1):
            gradient_weights = [0.0 for _ in range(len(self.weights))]
            gradient_bias = 0.0
            total_data_loss = 0.0

            for example in examples:
                transformed = self.transform_features(example.features)
                probability = self._sigmoid(self.bias + sum(weight * value for weight, value in zip(self.weights, transformed, strict=True)))
                total_data_loss += _binary_cross_entropy(probability, example.label)
                error = probability - example.label
                gradient_bias += error
                for index, value in enumerate(transformed):
                    gradient_weights[index] += error * value

            scale = learning_rate / sample_count
            for index in range(len(self.weights)):
                regularized_gradient = gradient_weights[index] + (sample_count * l2_lambda * self.weights[index])
                self.weights[index] -= scale * regularized_gradient
            self.bias -= scale * gradient_bias

            metrics = evaluate_classifier(self, examples, l2_lambda=l2_lambda)
            epoch_record = {
                "epoch": epoch,
                "loss": metrics.loss,
                "accuracy": metrics.accuracy,
                "f1_score": metrics.f1_score,
            }
            history.append(epoch_record)

            should_report = verbose and (epoch == 1 or epoch == epochs or epoch % max(1, report_every) == 0)
            if should_report:
                print(
                    f"Logistic epoch {epoch:>3}/{epochs} | "
                    f"loss {metrics.loss:.4f} | "
                    f"accuracy {metrics.accuracy:.2%} | "
                    f"f1 {metrics.f1_score:.2%}"
                )

        return LogisticRegressionTrainingResult(history=history, final_metrics=history and metrics or evaluate_classifier(self, examples, l2_lambda=l2_lambda))


class KNearestNeighboursClassifier:
    def __init__(self, neighbors: int = 7, distance_weighting: bool = True) -> None:
        if neighbors < 1:
            raise ValueError("neighbors must be at least 1.")
        self.neighbors = neighbors
        self.distance_weighting = distance_weighting
        self.examples: list[Example] = []

    def fit(self, examples: list[Example]) -> None:
        if not examples:
            raise ValueError("examples cannot be empty.")
        self.examples = [Example(features=example.features[:], label=example.label) for example in examples]

    @staticmethod
    def _distance(left: list[float], right: list[float]) -> float:
        return math.sqrt(sum((left_value - right_value) ** 2 for left_value, right_value in zip(left, right, strict=True)))

    def predict_probability(self, features: list[float]) -> float:
        if not self.examples:
            raise ValueError("fit must be called before predict_probability.")

        distances = [(self._distance(features, example.features), example.label) for example in self.examples]
        distances.sort(key=lambda item: item[0])
        nearest = distances[: min(self.neighbors, len(distances))]

        if nearest[0][0] == 0.0:
            return float(nearest[0][1])

        if self.distance_weighting:
            weighted_sum = 0.0
            total_weight = 0.0
            for distance, label in nearest:
                weight = 1.0 / max(distance, 1e-9)
                weighted_sum += weight * label
                total_weight += weight
            return weighted_sum / total_weight

        return sum(label for _, label in nearest) / len(nearest)

    def predict_label(self, features: list[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_probability(features) >= threshold else 0


class DecisionTreeClassifier:
    def __init__(self, max_depth: int = 6, min_samples_split: int = 8, min_impurity_decrease: float = 1e-4) -> None:
        if max_depth < 1:
            raise ValueError("max_depth must be at least 1.")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2.")
        if min_impurity_decrease < 0.0:
            raise ValueError("min_impurity_decrease cannot be negative.")

        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.min_impurity_decrease = min_impurity_decrease
        self.root: DecisionTreeNode | None = None
        self.input_size: int | None = None

    def fit(self, examples: list[Example]) -> None:
        if not examples:
            raise ValueError("examples cannot be empty.")
        self.input_size = len(examples[0].features)
        self.root = self._build_tree(examples, depth=0)

    @staticmethod
    def _gini_from_counts(positive_count: int, sample_count: int) -> float:
        if sample_count == 0:
            return 0.0
        positive_ratio = positive_count / sample_count
        negative_ratio = 1.0 - positive_ratio
        return 1.0 - (positive_ratio * positive_ratio) - (negative_ratio * negative_ratio)

    @classmethod
    def _gini_impurity(cls, examples: list[Example]) -> float:
        positive_count = sum(example.label for example in examples)
        return cls._gini_from_counts(positive_count, len(examples))

    def _best_split(self, examples: list[Example]) -> tuple[int | None, float | None, float]:
        sample_count = len(examples)
        if sample_count < 2:
            return None, None, float("inf")

        best_feature_index: int | None = None
        best_threshold: float | None = None
        best_impurity = float("inf")

        for feature_index in range(len(examples[0].features)):
            sorted_examples = sorted(examples, key=lambda example: example.features[feature_index])
            total_positive = sum(example.label for example in sorted_examples)
            left_positive = 0
            left_count = 0
            right_positive = total_positive
            right_count = sample_count

            for split_index in range(sample_count - 1):
                current_example = sorted_examples[split_index]
                left_count += 1
                left_positive += current_example.label
                right_count -= 1
                right_positive -= current_example.label

                current_value = current_example.features[feature_index]
                next_value = sorted_examples[split_index + 1].features[feature_index]
                if current_value == next_value:
                    continue

                left_impurity = self._gini_from_counts(left_positive, left_count)
                right_impurity = self._gini_from_counts(right_positive, right_count)
                weighted_impurity = ((left_count / sample_count) * left_impurity) + (
                    (right_count / sample_count) * right_impurity
                )

                if weighted_impurity < best_impurity:
                    best_impurity = weighted_impurity
                    best_feature_index = feature_index
                    best_threshold = (current_value + next_value) / 2.0

        return best_feature_index, best_threshold, best_impurity

    def _build_tree(self, examples: list[Example], depth: int) -> DecisionTreeNode:
        probability = sum(example.label for example in examples) / len(examples)
        node = DecisionTreeNode(probability=probability, sample_count=len(examples), depth=depth)

        current_impurity = self._gini_impurity(examples)
        if (
            depth >= self.max_depth
            or len(examples) < self.min_samples_split
            or current_impurity == 0.0
        ):
            return node

        feature_index, threshold, best_impurity = self._best_split(examples)
        if feature_index is None or threshold is None:
            return node

        impurity_drop = current_impurity - best_impurity
        if impurity_drop < self.min_impurity_decrease:
            return node

        left_examples = [example for example in examples if example.features[feature_index] <= threshold]
        right_examples = [example for example in examples if example.features[feature_index] > threshold]
        if not left_examples or not right_examples:
            return node

        node.feature_index = feature_index
        node.threshold = threshold
        node.left = self._build_tree(left_examples, depth + 1)
        node.right = self._build_tree(right_examples, depth + 1)
        return node

    def predict_probability(self, features: list[float]) -> float:
        if self.root is None:
            raise ValueError("fit must be called before predict_probability.")

        node = self.root
        while not node.is_leaf:
            if node.feature_index is None or node.threshold is None:
                break
            if features[node.feature_index] <= node.threshold:
                if node.left is None:
                    break
                node = node.left
            else:
                if node.right is None:
                    break
                node = node.right
        return node.probability

    def predict_label(self, features: list[float], threshold: float = 0.5) -> int:
        return 1 if self.predict_probability(features) >= threshold else 0

    def node_count(self) -> int:
        def count(node: DecisionTreeNode | None) -> int:
            if node is None:
                return 0
            return 1 + count(node.left) + count(node.right)

        return count(self.root)

    def leaf_count(self) -> int:
        def count(node: DecisionTreeNode | None) -> int:
            if node is None:
                return 0
            if node.is_leaf:
                return 1
            return count(node.left) + count(node.right)

        return count(self.root)


def evaluate_classifier(
    classifier: ProbabilisticBinaryClassifier,
    examples: list[Example],
    threshold: float = 0.5,
    l2_lambda: float = 0.0,
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
        probability = classifier.predict_probability(example.features)
        prediction = 1 if probability >= threshold else 0
        total_data_loss += _binary_cross_entropy(probability, example.label)
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

    regularization_loss = 0.0
    if l2_lambda > 0.0 and hasattr(classifier, "weight_l2_penalty"):
        regularization_loss = 0.5 * l2_lambda * getattr(classifier, "weight_l2_penalty")()

    precision = _safe_divide(true_positives, true_positives + false_positives)
    recall = _safe_divide(true_positives, true_positives + false_negatives)
    f1_score = _safe_divide(2 * precision * recall, precision + recall)
    average_data_loss = total_data_loss / len(examples)

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


def benchmark_classical_models(
    train_data: list[Example],
    test_data: list[Example],
    seed: int = 7,
    logistic_epochs: int = 220,
    logistic_learning_rate: float = 0.18,
    logistic_l2_lambda: float = 0.0005,
    knn_neighbors: int = 9,
    tree_max_depth: int = 6,
    tree_min_samples_split: int = 8,
    verbose: bool = True,
) -> ClassicalBenchmark:
    if not train_data:
        raise ValueError("train_data cannot be empty.")
    if not test_data:
        raise ValueError("test_data cannot be empty.")

    reports: list[ClassicalModelReport] = []

    logistic_model = LogisticRegressionClassifier(input_size=len(train_data[0].features), seed=seed, feature_mode="quadratic")
    logistic_result = logistic_model.fit(
        train_data,
        epochs=logistic_epochs,
        learning_rate=logistic_learning_rate,
        l2_lambda=logistic_l2_lambda,
        verbose=verbose,
    )
    reports.append(
        ClassicalModelReport(
            model_name="logistic_regression",
            train_metrics=logistic_result.final_metrics,
            test_metrics=evaluate_classifier(logistic_model, test_data, l2_lambda=logistic_l2_lambda),
            details={
                "feature_mode": logistic_model.feature_mode,
                "epochs": logistic_epochs,
                "learning_rate": logistic_learning_rate,
                "l2_lambda": logistic_l2_lambda,
                "engineered_feature_count": logistic_model.transformed_input_size,
            },
        )
    )

    knn_model = KNearestNeighboursClassifier(neighbors=knn_neighbors, distance_weighting=True)
    knn_model.fit(train_data)
    reports.append(
        ClassicalModelReport(
            model_name="k_nearest_neighbours",
            train_metrics=evaluate_classifier(knn_model, train_data),
            test_metrics=evaluate_classifier(knn_model, test_data),
            details={
                "neighbors": knn_neighbors,
                "distance_weighting": True,
            },
        )
    )

    tree_model = DecisionTreeClassifier(max_depth=tree_max_depth, min_samples_split=tree_min_samples_split)
    tree_model.fit(train_data)
    reports.append(
        ClassicalModelReport(
            model_name="decision_tree",
            train_metrics=evaluate_classifier(tree_model, train_data),
            test_metrics=evaluate_classifier(tree_model, test_data),
            details={
                "max_depth": tree_max_depth,
                "min_samples_split": tree_min_samples_split,
                "node_count": tree_model.node_count(),
                "leaf_count": tree_model.leaf_count(),
            },
        )
    )

    best_report = max(reports, key=lambda report: (report.test_metrics.accuracy, report.test_metrics.f1_score))
    return ClassicalBenchmark(reports=reports, best_model_name=best_report.model_name)


def save_classical_benchmark(path: str | Path, benchmark: ClassicalBenchmark) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(benchmark.to_dict(), indent=2), encoding="utf-8")
    return output_path
