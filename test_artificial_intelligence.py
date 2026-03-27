from __future__ import annotations

import unittest

from own_ai_model import (
    GridWorldConfig,
    GridWorldEnvironment,
    QLearningAgent,
    QLearningConfig,
    analyze_python_code,
    assess_ai_ethics,
    build_algorithm_plan,
)


class ArtificialIntelligenceTests(unittest.TestCase):
    def test_q_learning_gridworld_reaches_goal(self) -> None:
        environment = GridWorldEnvironment(
            GridWorldConfig(
                width=4,
                height=4,
                start=(0, 0),
                goal=(3, 3),
                walls=((1, 1),),
            )
        )
        agent = QLearningAgent(
            environment=environment,
            config=QLearningConfig(
                episodes=420,
                learning_rate=0.25,
                discount_factor=0.95,
                epsilon=0.95,
                epsilon_decay=0.992,
                minimum_epsilon=0.05,
                max_steps_per_episode=30,
                seed=9,
            ),
        )
        result = agent.train(verbose=False)
        self.assertGreaterEqual(result.success_rate, 0.70)
        self.assertEqual(result.greedy_path[0], (0, 0))
        self.assertEqual(result.greedy_path[-1], (3, 3))

    def test_algorithm_plan_matches_problem(self) -> None:
        plan = build_algorithm_plan("build a chatbot that answers user questions")
        self.assertEqual(plan.problem_type, "conversational_ai")
        self.assertTrue(any("fallback" in step.casefold() for step in plan.steps))

    def test_code_analysis_and_ethics_assessment(self) -> None:
        code = """
from math import *

def run(text):
    try:
        print("debug", eval(text))
    except:
        return None
"""
        suggestions = analyze_python_code(code)
        self.assertTrue(any(suggestion.kind == "security" for suggestion in suggestions))
        self.assertTrue(any(suggestion.kind == "reliability" for suggestion in suggestions))

        assessment = assess_ai_ethics("An AI hiring system stores face recordings and fully automated scores for candidates.")
        self.assertEqual(assessment.overall_risk, "high")
        self.assertIn("privacy", assessment.flagged_categories)
        self.assertIn("bias_and_fairness", assessment.flagged_categories)


if __name__ == "__main__":
    unittest.main()
