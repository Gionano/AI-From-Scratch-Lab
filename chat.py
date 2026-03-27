from __future__ import annotations

import argparse
from pathlib import Path

from own_ai_model import ChatSessionMemory, detect_runtime, format_runtime_info, generate_chatbot_reply, load_chatbot_model


EXIT_COMMANDS = {"exit", "quit", "keluar", "selesai"}


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with the local AI chatbot model.")
    parser.add_argument(
        "--model",
        default="artifacts/chatbot_model.json",
        help="Path to the trained chatbot model JSON file.",
    )
    parser.add_argument(
        "--message",
        help="Send one message and print one reply without opening interactive mode.",
    )
    parser.add_argument(
        "--show-intent",
        action="store_true",
        help="Show predicted intent and confidence together with the reply.",
    )
    parser.add_argument(
        "--show-runtime",
        action="store_true",
        help="Show backend/runtime information before chatting.",
    )
    args = parser.parse_args()

    model_path = Path(args.model)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Chatbot model was not found: {model_path.resolve()}. "
            "Run train_chatbot.py first."
        )

    classifier, responses_by_intent, fallback_responses, payload = load_chatbot_model(model_path)
    config_payload = payload.get("config", {})
    confidence_threshold = float(config_payload.get("confidence_threshold", 0.34))
    memory = ChatSessionMemory()
    runtime_info = detect_runtime()

    if args.show_runtime:
        print(format_runtime_info(runtime_info))
        print()

    if args.message:
        reply = generate_chatbot_reply(
            message=args.message,
            classifier=classifier,
            responses_by_intent=responses_by_intent,
            fallback_responses=fallback_responses,
            memory=memory,
            confidence_threshold=confidence_threshold,
        )
        if args.show_intent:
            print(f"[intent={reply.intent} confidence={reply.confidence:.2f}] {reply.reply}")
        else:
            print(reply.reply)
        return

    print("AI chat siap. Ketik pesanmu, lalu tekan Enter.")
    print("Ketik 'exit', 'quit', atau 'keluar' untuk berhenti.")

    while True:
        try:
            message = input("Kamu: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            print("AI: Sampai jumpa. Kalau mau lanjut, buka lagi chat ini.")
            return

        if not message:
            print("AI: Coba tulis sesuatu, nanti aku tanggapi.")
            continue
        if message.casefold() in EXIT_COMMANDS:
            print("AI: Oke, sampai nanti.")
            return

        reply = generate_chatbot_reply(
            message=message,
            classifier=classifier,
            responses_by_intent=responses_by_intent,
            fallback_responses=fallback_responses,
            memory=memory,
            confidence_threshold=confidence_threshold,
        )
        if args.show_intent:
            print(f"AI [{reply.intent} {reply.confidence:.2f}]: {reply.reply}")
        else:
            print(f"AI: {reply.reply}")


if __name__ == "__main__":
    main()
