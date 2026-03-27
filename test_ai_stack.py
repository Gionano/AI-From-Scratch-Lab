from __future__ import annotations

import unittest

from own_ai_model import (
    answer_ai_stack_question,
    build_ai_stack_payload,
    format_ai_concept_details,
    format_ai_stack_tree,
    match_ai_concept,
)


class AIStackTests(unittest.TestCase):
    def test_lookup_and_tree(self) -> None:
        machine_learning = match_ai_concept("machine learning")
        self.assertIsNotNone(machine_learning)
        self.assertEqual(machine_learning.name, "Machine Learning")

        tree = format_ai_stack_tree()
        self.assertIn("Artificial Intelligence", tree)
        self.assertIn("Small Language Models (SLM)", tree)

        payload = build_ai_stack_payload()
        self.assertTrue(payload)
        self.assertEqual(payload[0]["name"], "Artificial Intelligence")

    def test_concept_details_and_answers(self) -> None:
        details = format_ai_concept_details("slm")
        self.assertIn("Small Language Models (SLM)", details)
        self.assertIn("Path:", details)

        reply = answer_ai_stack_question("apa itu deep learning")
        self.assertIsNotNone(reply)
        self.assertIn("Deep Learning", reply)

        comparison = answer_ai_stack_question("apa beda ai dan machine learning")
        self.assertIsNotNone(comparison)
        self.assertIn("Machine Learning", comparison)


if __name__ == "__main__":
    unittest.main()
