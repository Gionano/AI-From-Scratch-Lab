from __future__ import annotations

import argparse

from own_ai_model import ai_stack_json, format_ai_concept_details, format_ai_stack_tree


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the AI -> ML -> NN -> DL -> GenAI -> SLM map in this project.")
    parser.add_argument(
        "--concept",
        help="Show detailed information for one concept, for example: 'machine learning' or 'slm'.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the concept tree as JSON.",
    )
    args = parser.parse_args()

    if args.concept:
        print(format_ai_concept_details(args.concept))
        return

    if args.json:
        print(ai_stack_json())
        return

    print(format_ai_stack_tree())


if __name__ == "__main__":
    main()
