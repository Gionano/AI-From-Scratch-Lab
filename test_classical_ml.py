from __future__ import annotations

from pathlib import Path
import unittest

from own_ai_model import (
    ArtifactConfig,
    DatasetConfig,
    DecisionTreeClassifier,
    KNearestNeighboursClassifier,
    LogisticRegressionClassifier,
    ModelConfig,
    ProjectConfig,
    TrainingConfig,
    benchmark_classical_models,
    build_datasets,
    evaluate_classifier,
    save_classical_benchmark,
)


class ClassicalMLTests(unittest.TestCase):
    def test_classical_models_learn_the_dataset(self) -> None:
        config = ProjectConfig(
            seed=9,
            dataset=DatasetConfig(
                train_samples=260,
                test_samples=120,
                coordinate_min=-1.0,
                coordinate_max=1.0,
                circle_radius=0.78,
                center_x=0.2,
                center_y=-0.1,
            ),
            model=ModelConfig(input_size=2, hidden_size=8, hidden_activation="relu"),
            training=TrainingConfig(
                epochs=160,
                learning_rate=0.12,
                batch_size=16,
                report_every=40,
                l2_lambda=0.0005,
            ),
            artifacts=ArtifactConfig(model_path="artifacts/test_unused.json"),
        )

        train_data, test_data = build_datasets(config)

        logistic_model = LogisticRegressionClassifier(input_size=2, seed=config.seed, feature_mode="quadratic")
        logistic_model.fit(
            train_data,
            epochs=180,
            learning_rate=0.18,
            l2_lambda=0.0005,
            verbose=False,
        )
        logistic_metrics = evaluate_classifier(logistic_model, test_data, l2_lambda=0.0005)
        self.assertGreaterEqual(logistic_metrics.accuracy, 0.85)

        knn_model = KNearestNeighboursClassifier(neighbors=9, distance_weighting=True)
        knn_model.fit(train_data)
        knn_metrics = evaluate_classifier(knn_model, test_data)
        self.assertGreaterEqual(knn_metrics.accuracy, 0.90)

        tree_model = DecisionTreeClassifier(max_depth=6, min_samples_split=8)
        tree_model.fit(train_data)
        tree_metrics = evaluate_classifier(tree_model, test_data)
        self.assertGreaterEqual(tree_metrics.accuracy, 0.84)
        self.assertGreater(tree_model.node_count(), 1)
        self.assertGreater(tree_model.leaf_count(), 1)

    def test_benchmark_and_report_export(self) -> None:
        config = ProjectConfig(
            seed=13,
            dataset=DatasetConfig(
                train_samples=220,
                test_samples=100,
                coordinate_min=-1.0,
                coordinate_max=1.0,
                circle_radius=0.78,
                center_x=0.2,
                center_y=-0.1,
            ),
            model=ModelConfig(input_size=2, hidden_size=8, hidden_activation="relu"),
            training=TrainingConfig(
                epochs=120,
                learning_rate=0.12,
                batch_size=12,
                report_every=40,
                l2_lambda=0.0005,
            ),
            artifacts=ArtifactConfig(model_path="artifacts/test_unused.json"),
        )

        train_data, test_data = build_datasets(config)
        benchmark = benchmark_classical_models(
            train_data=train_data,
            test_data=test_data,
            seed=config.seed,
            logistic_epochs=120,
            verbose=False,
        )

        self.assertEqual(len(benchmark.reports), 3)
        self.assertIn(benchmark.best_model_name, {"logistic_regression", "k_nearest_neighbours", "decision_tree"})

        output_path = Path("artifacts") / "test_classical_report.json"
        try:
            saved_path = save_classical_benchmark(output_path, benchmark)
            self.assertTrue(saved_path.exists())
            self.assertIn("reports", benchmark.to_dict())
        finally:
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
