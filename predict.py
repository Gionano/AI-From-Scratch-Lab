from __future__ import annotations

import argparse
import json
from pathlib import Path

from own_ai_model import load_model, load_prediction_points, predict_points, save_predictions_csv


def main() -> None:
    parser = argparse.ArgumentParser(description="Use the trained model to classify a single 2D point.")
    parser.add_argument("x", type=float, nargs="?", help="First feature value.")
    parser.add_argument("y", type=float, nargs="?", help="Second feature value.")
    parser.add_argument(
        "--model",
        default="artifacts/model_params.json",
        help="Path to the trained model parameters JSON file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Classification threshold used for predicted_class.",
    )
    parser.add_argument(
        "--input-file",
        help="Optional CSV file with multiple points to predict.",
    )
    parser.add_argument(
        "--output-file",
        help="Optional CSV path to save batch predictions.",
    )
    parser.add_argument(
        "--show-forward-pass",
        action="store_true",
        help="Show explicit forward pass values using y = f(Wx + b).",
    )
    parser.add_argument(
        "--show-loss",
        action="store_true",
        help="Show loss breakdown. Requires --label when predicting a single point.",
    )
    parser.add_argument(
        "--label",
        type=int,
        choices=[0, 1],
        help="True label for loss calculation when predicting a single point.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(f"Model file was not found: {model_path.resolve()}")

    model, payload = load_model(model_path)
    final_metrics = payload.get("final_metrics", {})
    training_summary = payload.get("training_summary", {})

    if args.input_file:
        if args.x is not None or args.y is not None:
            raise ValueError("Use either positional x/y inputs or --input-file, not both.")

        points = load_prediction_points(args.input_file)
        predictions = predict_points(model, points, threshold=args.threshold)
        positive_predictions = sum(record.predicted_class for record in predictions)

        print(f"Model file: {model_path.resolve()}")
        print(f"Processed rows: {len(predictions)}")
        print(f"Threshold: {args.threshold:.2f}")
        print(f"Predicted class 1 rows: {positive_predictions}")
        print(f"Predicted class 0 rows: {len(predictions) - positive_predictions}")

        preview_count = min(5, len(predictions))
        print("Preview:")
        for record in predictions[:preview_count]:
            print(
                f"  row {record.row_number}: "
                f"x={record.x:.4f}, y={record.y:.4f}, "
                f"prob={record.probability:.4f}, class={record.predicted_class}"
            )

        if args.output_file:
            saved_path = save_predictions_csv(args.output_file, predictions)
            print(f"Batch predictions saved to: {saved_path.resolve()}")
        return

    if args.x is None or args.y is None:
        raise ValueError("Provide x and y values, or use --input-file for batch prediction.")

    probability = model.predict_probability([args.x, args.y])
    prediction = 1 if probability >= args.threshold else 0
    forward_result = model.forward_pass([args.x, args.y])

    print(f"Model file: {model_path.resolve()}")
    print(f"Input point: [{args.x}, {args.y}]")
    print(f"Threshold: {args.threshold:.2f}")
    print(f"Predicted probability of class 1: {probability:.4f}")
    print(f"Predicted class: {prediction}")
    if args.show_forward_pass:
        print("Forward Pass:")
        print("  hidden_z = W1x + b1")
        print(f"  hidden_z values: {json.dumps([round(value, 6) for value in forward_result.hidden_linear])}")
        print(f"  hidden_a = {model.hidden_activation_name}(hidden_z)")
        print(f"  hidden_a values: {json.dumps([round(value, 6) for value in forward_result.hidden_activation])}")
        print("  output_z = W2h + b2")
        print(f"  output_z value: {forward_result.output_linear:.6f}")
        print("  y = sigmoid(output_z)")
        print(f"  y value: {forward_result.output_activation:.6f}")
    if args.show_loss:
        if args.label is None:
            raise ValueError("--show-loss requires --label 0 or 1.")
        loss_function_name = "binary_cross_entropy"
        if isinstance(payload.get("config"), dict):
            training_config = payload["config"].get("training", {})
            if isinstance(training_config, dict):
                loss_function_name = str(training_config.get("loss_function", loss_function_name))
        loss = model.loss_from_features([args.x, args.y], args.label, loss_function_name=loss_function_name)
        print("Loss Function:")
        print(f"  loss_type: {loss.loss_function}")
        print(f"  true_label: {loss.label}")
        print(f"  prediction: {loss.prediction:.6f}")
        print(f"  data_loss: {loss.data_loss:.6f}")
        print(f"  regularization_loss: {loss.regularization_loss:.6f}")
        print(f"  total_loss: {loss.total_loss:.6f}")
    if final_metrics:
        print(
            "Stored test accuracy:",
            f"{final_metrics.get('test', {}).get('accuracy', 0.0):.2%}",
        )
    if training_summary:
        print(
            "Best epoch used:",
            training_summary.get("best_epoch", "unknown"),
        )


if __name__ == "__main__":
    main()
