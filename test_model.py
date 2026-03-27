from __future__ import annotations

import csv
from pathlib import Path
import unittest

from own_ai_model import (
    ArtifactConfig,
    DatasetConfig,
    ForwardPassResult,
    LossBreakdown,
    ModelConfig,
    ProjectConfig,
    SimpleNeuralNetwork,
    TrainingConfig,
    build_datasets,
    evaluate_model,
    load_prediction_points,
    load_model,
    predict_points,
    save_history_csv,
    save_model,
    save_predictions_csv,
    train_model,
)


class ModelTrainingTests(unittest.TestCase):
    def test_training_and_reload_are_stable(self) -> None:
        config = ProjectConfig(
            seed=11,
            dataset=DatasetConfig(
                train_samples=320,
                test_samples=120,
                coordinate_min=-1.0,
                coordinate_max=1.0,
                circle_radius=0.78,
                center_x=0.2,
                center_y=-0.1,
            ),
            model=ModelConfig(input_size=2, hidden_size=10, hidden_activation="relu"),
            training=TrainingConfig(
                epochs=200,
                learning_rate=0.12,
                batch_size=16,
                report_every=50,
                loss_function="binary_cross_entropy",
                optimization_method="gradient_descent",
                early_stopping_patience=20,
                early_stopping_min_delta=0.0005,
                l2_lambda=0.001,
                gradient_clip_value=2.0,
                learning_rate_decay=0.01,
                momentum=0.75,
                mistake_focus_power=1.1,
                bias_learning_rate_multiplier=1.05,
            ),
            artifacts=ArtifactConfig(model_path="artifacts/test_model.json"),
        )

        train_data, test_data = build_datasets(config)
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
            verbose=False,
        )

        test_metrics = evaluate_model(model, test_data)
        self.assertGreaterEqual(test_metrics.accuracy, 0.88)
        self.assertGreaterEqual(test_metrics.f1_score, 0.88)
        self.assertLessEqual(training_result.best_epoch, training_result.epochs_ran)
        self.assertEqual(training_result.start_epoch, 0)
        self.assertEqual(training_result.final_epoch, training_result.epochs_ran)
        self.assertGreater(training_result.final_parameter_stats.parameter_count, 0)
        self.assertGreater(training_result.final_parameter_stats.l2_norm, 0.0)
        self.assertGreaterEqual(training_result.final_velocity_norm, 0.0)

        path = Path("artifacts") / "test_model_temp.json"
        try:
            save_model(
                path=path,
                model=model,
                config=config,
                history=training_result.history,
                final_metrics={
                    "test": test_metrics.to_dict(),
                    "train": evaluate_model(model, train_data).to_dict(),
                },
                training_summary=training_result.to_dict(),
            )
            reloaded_model, payload = load_model(path)
            self.assertIn("training_summary", payload)
            self.assertEqual(payload["training_summary"]["final_epoch"], training_result.final_epoch)
            self.assertIn("final_parameter_stats", payload["training_summary"])

            for example in test_data[:10]:
                original = model.predict_probability(example.features)
                restored = reloaded_model.predict_probability(example.features)
                self.assertAlmostEqual(original, restored, places=12)
        finally:
            path.unlink(missing_ok=True)

    def test_invalid_training_config_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=0, learning_rate=0.1, batch_size=16, report_every=5)
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=10, learning_rate=0.1, batch_size=16, report_every=5, l2_lambda=-0.1)
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=10, learning_rate=0.1, batch_size=16, report_every=5, momentum=1.0)
        with self.assertRaises(ValueError):
            TrainingConfig(epochs=10, learning_rate=0.1, batch_size=16, report_every=5, mistake_focus_power=0.8)
        with self.assertRaises(ValueError):
            ModelConfig(input_size=2, hidden_size=4, hidden_activation="softplus")
        with self.assertRaises(ValueError):
            TrainingConfig(
                epochs=10,
                learning_rate=0.1,
                batch_size=16,
                report_every=5,
                loss_function="hinge",
            )

    def test_forward_pass_and_loss_function_are_explicit(self) -> None:
        model = SimpleNeuralNetwork(input_size=2, hidden_size=4, seed=3, hidden_activation="relu")
        features = [0.1, -0.2]

        forward_result = model.forward_pass(features)
        self.assertIsInstance(forward_result, ForwardPassResult)
        self.assertEqual(forward_result.input_vector, features)
        self.assertEqual(len(forward_result.hidden_linear), 4)
        self.assertEqual(len(forward_result.hidden_activation), 4)
        self.assertTrue(all(value >= 0.0 for value in forward_result.hidden_activation))
        self.assertGreaterEqual(forward_result.output_activation, 0.0)
        self.assertLessEqual(forward_result.output_activation, 1.0)

        loss_breakdown = model.loss_from_features(
            features,
            label=1,
            loss_function_name="binary_cross_entropy",
            l2_lambda=0.001,
        )
        self.assertIsInstance(loss_breakdown, LossBreakdown)
        self.assertEqual(loss_breakdown.loss_function, "binary_cross_entropy")
        self.assertGreater(loss_breakdown.total_loss, 0.0)
        self.assertGreaterEqual(loss_breakdown.regularization_loss, 0.0)

        sigmoid_model = SimpleNeuralNetwork(input_size=2, hidden_size=4, seed=3, hidden_activation="sigmoid")
        sigmoid_forward = sigmoid_model.forward_pass(features)
        self.assertTrue(all(0.0 <= value <= 1.0 for value in sigmoid_forward.hidden_activation))
        mse_breakdown = sigmoid_model.loss_from_features(
            features,
            label=1,
            loss_function_name="mean_squared_error",
            l2_lambda=0.001,
        )
        self.assertEqual(mse_breakdown.loss_function, "mean_squared_error")
        self.assertGreaterEqual(mse_breakdown.total_loss, mse_breakdown.data_loss)

    def test_history_and_batch_prediction_exports(self) -> None:
        config = ProjectConfig(
            seed=5,
            dataset=DatasetConfig(
                train_samples=120,
                test_samples=60,
                coordinate_min=-1.0,
                coordinate_max=1.0,
                circle_radius=0.78,
                center_x=0.2,
                center_y=-0.1,
            ),
            model=ModelConfig(input_size=2, hidden_size=8, hidden_activation="sigmoid"),
            training=TrainingConfig(
                epochs=40,
                learning_rate=0.12,
                batch_size=12,
                report_every=20,
                loss_function="mean_squared_error",
                optimization_method="gradient_descent",
                l2_lambda=0.001,
                gradient_clip_value=1.5,
                learning_rate_decay=0.02,
                momentum=0.7,
                mistake_focus_power=1.2,
                bias_learning_rate_multiplier=1.08,
            ),
            artifacts=ArtifactConfig(model_path="artifacts/test_model.json"),
        )

        train_data, test_data = build_datasets(config)
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
            epoch_offset=10,
            verbose=False,
        )

        history_path = Path("artifacts") / "test_history.csv"
        input_path = Path("artifacts") / "test_predict_input.csv"
        output_path = Path("artifacts") / "test_predict_output.csv"
        try:
            saved_history_path = save_history_csv(history_path, training_result.history)
            self.assertTrue(saved_history_path.exists())
            self.assertEqual(training_result.history[0]["epoch"], 11)
            self.assertGreater(training_result.history[0]["learning_rate"], training_result.history[-1]["learning_rate"])
            self.assertIn("parameter_l2_norm", training_result.history[0])
            self.assertIn("mean_mistake_signal", training_result.history[0])
            self.assertIn("velocity_norm", training_result.history[0])

            input_path.write_text("x,y\n0.1,0.2\n-0.9,-0.9\n", encoding="utf-8")
            points = load_prediction_points(input_path)
            predictions = predict_points(model, points, threshold=0.5)
            saved_predictions_path = save_predictions_csv(output_path, predictions)
            self.assertTrue(saved_predictions_path.exists())
            self.assertEqual(len(predictions), 2)

            with output_path.open("r", encoding="utf-8", newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(len(rows), 2)
            self.assertIn("probability", rows[0])
        finally:
            history_path.unlink(missing_ok=True)
            input_path.unlink(missing_ok=True)
            output_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
