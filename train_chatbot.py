from __future__ import annotations

import argparse
from pathlib import Path

from own_ai_model import (
    detect_runtime,
    format_runtime_info,
    load_chatbot_config,
    load_intent_dataset,
    save_chatbot_model,
    train_intent_classifier,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a simple local chatbot intent model.")
    parser.add_argument(
        "--config",
        default="config/chatbot_config.json",
        help="Path to the chatbot training config JSON file.",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Reduce training logs.",
    )
    args = parser.parse_args()

    config = load_chatbot_config(args.config)
    runtime_info = detect_runtime()
    examples, responses_by_intent, fallback_responses = load_intent_dataset(config.data_path)
    classifier, training_result = train_intent_classifier(
        examples=examples,
        config=config,
        verbose=not args.quiet,
    )
    output_path = save_chatbot_model(
        path=config.model_path,
        classifier=classifier,
        config=config,
        responses_by_intent=responses_by_intent,
        fallback_responses=fallback_responses,
        training_result=training_result,
        runtime_info=runtime_info,
    )

    print()
    print(format_runtime_info(runtime_info))
    print()
    print(f"Chatbot training complete. Model saved to: {Path(output_path).resolve()}")
    print(f"Training accuracy: {training_result.accuracy:.2%}")
    print(f"Final loss: {training_result.loss:.4f}")
    print(f"Training phrases: {len(examples)}")
    print(f"Known intents: {len(responses_by_intent)}")


if __name__ == "__main__":
    main()
