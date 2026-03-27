from __future__ import annotations

from own_ai_model import detect_runtime, format_runtime_info


def main() -> None:
    print(format_runtime_info(detect_runtime()))


if __name__ == "__main__":
    main()
