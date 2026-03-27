from __future__ import annotations

import argparse
from pathlib import Path

from own_ai_model import (
    GridWorldConfig,
    GridWorldEnvironment,
    QLearningAgent,
    QLearningConfig,
    analyze_python_file,
    assess_ai_ethics,
    build_algorithm_plan,
    format_algorithm_plan,
    format_code_suggestions,
    format_ethics_assessment,
)


def run_reinforcement_learning_demo() -> None:
    environment = GridWorldEnvironment(
        GridWorldConfig(
            width=5,
            height=5,
            start=(0, 0),
            goal=(4, 4),
            walls=((1, 1), (1, 2), (3, 3)),
        )
    )
    agent = QLearningAgent(
        environment=environment,
        config=QLearningConfig(
            episodes=500,
            learning_rate=0.22,
            discount_factor=0.95,
            epsilon=0.95,
            epsilon_decay=0.992,
        ),
    )
    result = agent.train(verbose=False)

    print("Reinforcement Learning Demo")
    print(f"Success rate: {result.success_rate:.2%}")
    print(f"Average reward: {result.average_reward:.4f}")
    print(f"Greedy path: {result.greedy_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Explore Artificial Intelligence foundations in this project.")
    parser.add_argument(
        "--rl",
        action="store_true",
        help="Run a small reinforcement-learning demo with Q-learning in GridWorld.",
    )
    parser.add_argument(
        "--plan",
        help="Build an algorithm plan from a problem statement.",
    )
    parser.add_argument(
        "--analyze-code",
        help="Analyze a Python file and print augmented-programming suggestions.",
    )
    parser.add_argument(
        "--ethics",
        help="Assess AI ethics risks from a short system description.",
    )
    args = parser.parse_args()

    ran_anything = False

    if args.rl:
        run_reinforcement_learning_demo()
        ran_anything = True

    if args.plan:
        if ran_anything:
            print()
        print(format_algorithm_plan(build_algorithm_plan(args.plan)))
        ran_anything = True

    if args.analyze_code:
        if ran_anything:
            print()
        suggestions = analyze_python_file(args.analyze_code)
        print(f"Code file: {Path(args.analyze_code).resolve()}")
        print(format_code_suggestions(suggestions))
        ran_anything = True

    if args.ethics:
        if ran_anything:
            print()
        print(format_ethics_assessment(assess_ai_ethics(args.ethics)))
        ran_anything = True

    if not ran_anything:
        parser.print_help()


if __name__ == "__main__":
    main()
