from __future__ import annotations

from pathlib import Path
import unittest

from own_ai_model import (
    ChatSessionMemory,
    ChatbotTrainingConfig,
    detect_runtime,
    generate_chatbot_reply,
    load_chatbot_model,
    load_intent_dataset,
    save_chatbot_model,
    train_intent_classifier,
)


class ChatbotTests(unittest.TestCase):
    def test_chatbot_training_and_memory(self) -> None:
        artifact_path = Path("artifacts") / "test_chatbot_model.json"
        config = ChatbotTrainingConfig(
            data_path="chatbot_data/intents_id.json",
            model_path=str(artifact_path),
            epochs=250,
            learning_rate=0.9,
            report_every=100,
            min_word_frequency=1,
            confidence_threshold=0.28,
            seed=17,
        )

        examples, responses_by_intent, fallback_responses = load_intent_dataset(config.data_path)
        classifier, training_result = train_intent_classifier(examples, config, verbose=False)
        self.assertGreaterEqual(training_result.accuracy, 0.90)
        runtime_info = detect_runtime()

        try:
            save_chatbot_model(
                path=artifact_path,
                classifier=classifier,
                config=config,
                responses_by_intent=responses_by_intent,
                fallback_responses=fallback_responses,
                training_result=training_result,
                runtime_info=runtime_info,
            )

            reloaded_classifier, reloaded_responses, reloaded_fallbacks, payload = load_chatbot_model(artifact_path)
            self.assertIn("training_result", payload)
            self.assertIn("runtime", payload)
            self.assertIn(payload["runtime"]["backend"], {"python", "torch"})

            memory = ChatSessionMemory()
            remember_name = generate_chatbot_reply(
                message="nama saya budi",
                classifier=reloaded_classifier,
                responses_by_intent=reloaded_responses,
                fallback_responses=reloaded_fallbacks,
                memory=memory,
                confidence_threshold=config.confidence_threshold,
            )
            ask_name = generate_chatbot_reply(
                message="siapa nama saya",
                classifier=reloaded_classifier,
                responses_by_intent=reloaded_responses,
                fallback_responses=reloaded_fallbacks,
                memory=memory,
                confidence_threshold=config.confidence_threshold,
            )
            coding_reply = generate_chatbot_reply(
                message="program saya error",
                classifier=reloaded_classifier,
                responses_by_intent=reloaded_responses,
                fallback_responses=reloaded_fallbacks,
                memory=memory,
                confidence_threshold=config.confidence_threshold,
            )
            memory_note = generate_chatbot_reply(
                message="ingat bahwa saya suka matematika",
                classifier=reloaded_classifier,
                responses_by_intent=reloaded_responses,
                fallback_responses=reloaded_fallbacks,
                memory=memory,
                confidence_threshold=config.confidence_threshold,
            )
            memory_summary = generate_chatbot_reply(
                message="apa yang kamu ingat tentang saya",
                classifier=reloaded_classifier,
                responses_by_intent=reloaded_responses,
                fallback_responses=reloaded_fallbacks,
                memory=memory,
                confidence_threshold=config.confidence_threshold,
            )
            ai_knowledge = generate_chatbot_reply(
                message="apa itu machine learning",
                classifier=reloaded_classifier,
                responses_by_intent=reloaded_responses,
                fallback_responses=reloaded_fallbacks,
                memory=memory,
                confidence_threshold=config.confidence_threshold,
            )

            self.assertEqual(remember_name.intent, "memory")
            self.assertIn("Budi", ask_name.reply)
            self.assertEqual(coding_reply.intent, "coding_help")
            self.assertIn("matematika", memory_note.reply.casefold())
            self.assertIn("Budi", memory_summary.reply)
            self.assertEqual(ai_knowledge.intent, "ai_knowledge")
            self.assertIn("Machine Learning", ai_knowledge.reply)
        finally:
            artifact_path.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
