from __future__ import annotations

import ast
from dataclasses import dataclass
import math
from pathlib import Path
import random
import re


ACTIONS = ("up", "down", "left", "right")


@dataclass(frozen=True)
class GridWorldConfig:
    width: int = 4
    height: int = 4
    start: tuple[int, int] = (0, 0)
    goal: tuple[int, int] = (3, 3)
    walls: tuple[tuple[int, int], ...] = ()
    step_reward: float = -0.04
    goal_reward: float = 1.0
    wall_penalty: float = -0.2

    def __post_init__(self) -> None:
        if self.width < 2 or self.height < 2:
            raise ValueError("GridWorld width and height must be at least 2.")
        if not (0 <= self.start[0] < self.width and 0 <= self.start[1] < self.height):
            raise ValueError("GridWorld start must be inside the grid.")
        if not (0 <= self.goal[0] < self.width and 0 <= self.goal[1] < self.height):
            raise ValueError("GridWorld goal must be inside the grid.")
        if self.start == self.goal:
            raise ValueError("GridWorld start and goal must be different.")
        if self.goal in self.walls or self.start in self.walls:
            raise ValueError("Walls cannot overlap start or goal.")


@dataclass(frozen=True)
class QLearningConfig:
    episodes: int = 450
    learning_rate: float = 0.2
    discount_factor: float = 0.95
    epsilon: float = 0.9
    epsilon_decay: float = 0.992
    minimum_epsilon: float = 0.05
    max_steps_per_episode: int = 40
    seed: int = 7

    def __post_init__(self) -> None:
        if self.episodes < 1:
            raise ValueError("episodes must be at least 1.")
        if not 0.0 < self.learning_rate <= 1.0:
            raise ValueError("learning_rate must be between 0 and 1.")
        if not 0.0 <= self.discount_factor <= 1.0:
            raise ValueError("discount_factor must be between 0 and 1.")
        if not 0.0 <= self.epsilon <= 1.0:
            raise ValueError("epsilon must be between 0 and 1.")
        if not 0.0 < self.epsilon_decay <= 1.0:
            raise ValueError("epsilon_decay must be between 0 and 1.")
        if not 0.0 <= self.minimum_epsilon <= 1.0:
            raise ValueError("minimum_epsilon must be between 0 and 1.")
        if self.max_steps_per_episode < 1:
            raise ValueError("max_steps_per_episode must be at least 1.")


@dataclass(frozen=True)
class ReinforcementLearningResult:
    history: list[dict[str, float | int | bool]]
    success_rate: float
    average_reward: float
    greedy_path: list[tuple[int, int]]
    policy: dict[str, str]

    def to_dict(self) -> dict[str, object]:
        return {
            "history": self.history,
            "success_rate": self.success_rate,
            "average_reward": self.average_reward,
            "greedy_path": [list(position) for position in self.greedy_path],
            "policy": self.policy,
        }


@dataclass(frozen=True)
class AlgorithmPlan:
    title: str
    problem_type: str
    inputs: list[str]
    outputs: list[str]
    steps: list[str]
    notes: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "title": self.title,
            "problem_type": self.problem_type,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "steps": self.steps,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class CodeSuggestion:
    kind: str
    severity: str
    line: int
    message: str
    suggestion: str

    def to_dict(self) -> dict[str, object]:
        return {
            "kind": self.kind,
            "severity": self.severity,
            "line": self.line,
            "message": self.message,
            "suggestion": self.suggestion,
        }


@dataclass(frozen=True)
class EthicsAssessment:
    overall_risk: str
    risks: list[str]
    mitigations: list[str]
    flagged_categories: list[str]

    def to_dict(self) -> dict[str, object]:
        return {
            "overall_risk": self.overall_risk,
            "risks": self.risks,
            "mitigations": self.mitigations,
            "flagged_categories": self.flagged_categories,
        }


class GridWorldEnvironment:
    def __init__(self, config: GridWorldConfig) -> None:
        self.config = config
        self.state = config.start

    def reset(self) -> tuple[int, int]:
        self.state = self.config.start
        return self.state

    def is_terminal(self, state: tuple[int, int]) -> bool:
        return state == self.config.goal

    def _candidate_state(self, state: tuple[int, int], action: str) -> tuple[int, int]:
        x_position, y_position = state
        if action == "up":
            y_position -= 1
        elif action == "down":
            y_position += 1
        elif action == "left":
            x_position -= 1
        elif action == "right":
            x_position += 1
        else:
            raise ValueError(f"Unsupported action: {action}")
        return x_position, y_position

    def step(self, action: str) -> tuple[tuple[int, int], float, bool]:
        next_state = self._candidate_state(self.state, action)
        x_position, y_position = next_state
        if not (0 <= x_position < self.config.width and 0 <= y_position < self.config.height):
            return self.state, self.config.wall_penalty, False
        if next_state in self.config.walls:
            return self.state, self.config.wall_penalty, False

        self.state = next_state
        if self.state == self.config.goal:
            return self.state, self.config.goal_reward, True
        return self.state, self.config.step_reward, False

    def all_states(self) -> list[tuple[int, int]]:
        states: list[tuple[int, int]] = []
        for y_position in range(self.config.height):
            for x_position in range(self.config.width):
                state = (x_position, y_position)
                if state not in self.config.walls:
                    states.append(state)
        return states


class QLearningAgent:
    def __init__(self, environment: GridWorldEnvironment, config: QLearningConfig) -> None:
        self.environment = environment
        self.config = config
        self.rng = random.Random(config.seed)
        self.q_table: dict[tuple[int, int], dict[str, float]] = {
            state: {action: 0.0 for action in ACTIONS}
            for state in environment.all_states()
        }

    def _best_action(self, state: tuple[int, int]) -> str:
        action_values = self.q_table[state]
        return max(ACTIONS, key=lambda action: (action_values[action], action))

    def _epsilon_greedy_action(self, state: tuple[int, int], epsilon: float) -> str:
        if self.rng.random() < epsilon:
            return self.rng.choice(ACTIONS)
        return self._best_action(state)

    def train(self, verbose: bool = True) -> ReinforcementLearningResult:
        history: list[dict[str, float | int | bool]] = []
        epsilon = self.config.epsilon
        successes = 0
        rewards: list[float] = []

        for episode in range(1, self.config.episodes + 1):
            state = self.environment.reset()
            episode_reward = 0.0
            reached_goal = False

            for step in range(1, self.config.max_steps_per_episode + 1):
                action = self._epsilon_greedy_action(state, epsilon)
                next_state, reward, done = self.environment.step(action)

                best_next_value = 0.0 if done else max(self.q_table[next_state].values())
                old_value = self.q_table[state][action]
                self.q_table[state][action] = old_value + self.config.learning_rate * (
                    reward + (self.config.discount_factor * best_next_value) - old_value
                )

                episode_reward += reward
                state = next_state
                if done:
                    reached_goal = True
                    successes += 1
                    break

            history.append(
                {
                    "episode": episode,
                    "reward": episode_reward,
                    "steps": step,
                    "reached_goal": reached_goal,
                    "epsilon": epsilon,
                }
            )
            rewards.append(episode_reward)
            epsilon = max(self.config.minimum_epsilon, epsilon * self.config.epsilon_decay)

            should_report = verbose and (episode == 1 or episode == self.config.episodes or episode % 50 == 0)
            if should_report:
                print(
                    f"RL episode {episode:>3}/{self.config.episodes} | "
                    f"reward {episode_reward:.3f} | "
                    f"steps {step} | "
                    f"goal {reached_goal} | "
                    f"epsilon {epsilon:.3f}"
                )

        greedy_path = self.greedy_path()
        policy = {
            f"{state[0]},{state[1]}": self._best_action(state)
            for state in self.q_table
            if not self.environment.is_terminal(state)
        }
        return ReinforcementLearningResult(
            history=history,
            success_rate=successes / self.config.episodes,
            average_reward=sum(rewards) / len(rewards),
            greedy_path=greedy_path,
            policy=policy,
        )

    def greedy_path(self, max_steps: int | None = None) -> list[tuple[int, int]]:
        path = [self.environment.config.start]
        state = self.environment.config.start
        limit = max_steps or (self.environment.config.width * self.environment.config.height * 2)
        seen: set[tuple[int, int]] = set()
        previous_state = self.environment.state
        try:
            self.environment.state = state
            for _ in range(limit):
                if state == self.environment.config.goal:
                    break
                action = self._best_action(state)
                next_state, _, done = self.environment.step(action)
                if next_state == state and state in seen:
                    break
                path.append(next_state)
                seen.add(state)
                state = next_state
                if done:
                    break
        finally:
            self.environment.state = previous_state
        return path


def _normalize_problem(problem_statement: str) -> str:
    lowered = problem_statement.casefold()
    return re.sub(r"\s+", " ", lowered).strip()


def build_algorithm_plan(problem_statement: str) -> AlgorithmPlan:
    normalized = _normalize_problem(problem_statement)
    if not normalized:
        raise ValueError("problem_statement cannot be empty.")

    problem_type = "general_problem_solving"
    inputs = ["problem statement", "constraints", "example cases"]
    outputs = ["working solution", "verification result"]
    steps = [
        "Clarify the target, constraints, and success criteria.",
        "List the inputs, outputs, and edge cases.",
        "Choose a baseline algorithm or strategy that is easy to test.",
        "Implement the smallest correct version first.",
        "Evaluate the result against examples and failure cases.",
        "Refine the algorithm for speed, accuracy, or maintainability.",
    ]
    notes = ["Mulai dari solusi yang sederhana, lalu tingkatkan secara bertahap."]

    if any(keyword in normalized for keyword in ("classify", "classification", "label", "spam", "sentiment", "deteksi", "prediksi kelas")):
        problem_type = "classification"
        inputs = ["labeled training data", "feature columns", "evaluation split"]
        outputs = ["predicted class", "accuracy and F1 score"]
        steps = [
            "Collect or clean labeled examples for each class.",
            "Split the data into train and test sets.",
            "Build useful features or representations from the inputs.",
            "Train a baseline classifier and measure accuracy and F1 score.",
            "Inspect mistakes and improve features, thresholds, or model choice.",
            "Deploy the classifier together with confidence checks.",
        ]
        notes = ["Cocok untuk spam detection, intent classification, dan binary prediction."]
    elif any(keyword in normalized for keyword in ("regression", "price", "forecast", "nilai", "house", "temperature")):
        problem_type = "regression"
        inputs = ["numeric target values", "feature columns", "train/test split"]
        outputs = ["predicted numeric value", "error metrics"]
        steps = [
            "Collect historical examples with numeric targets.",
            "Normalize or scale the important features when needed.",
            "Train a simple regression baseline first.",
            "Measure MAE, MSE, or another relevant error metric.",
            "Inspect outliers and improve the feature set.",
            "Compare the baseline with a stronger model only after the baseline is stable.",
        ]
        notes = ["Gunakan baseline sederhana lebih dulu sebelum model yang lebih kompleks."]
    elif any(keyword in normalized for keyword in ("chat", "dialog", "conversation", "assistant", "chatbot")):
        problem_type = "conversational_ai"
        inputs = ["user messages", "response patterns or training data", "conversation memory"]
        outputs = ["reply", "intent or reasoning trace"]
        steps = [
            "Define the kinds of user requests the assistant must handle.",
            "Decide whether the system should be rule-based, intent-based, or generative.",
            "Add memory rules for names, preferences, or session context.",
            "Prepare fallback behavior for unknown or unsafe requests.",
            "Test common conversations, edge cases, and recovery paths.",
            "Measure reply quality and expand the dataset or reasoning tools over time.",
        ]
        notes = ["Untuk assistant lokal, mulai dari intent-based lalu naik ke model yang lebih besar."]
    elif any(keyword in normalized for keyword in ("path", "maze", "route", "robot", "reward", "agent", "game")):
        problem_type = "reinforcement_learning"
        inputs = ["environment states", "actions", "reward design"]
        outputs = ["policy", "best path or action strategy"]
        steps = [
            "Define the environment, state space, and valid actions.",
            "Design rewards so the agent is encouraged toward the right behavior.",
            "Train a simple RL agent such as Q-learning first.",
            "Track reward, success rate, and greedy policy over time.",
            "Inspect failure states and adjust exploration or rewards.",
            "Only move to deep RL after the tabular version is stable.",
        ]
        notes = ["Reward design sangat menentukan perilaku agent."]

    title = problem_statement.strip().rstrip(".")
    return AlgorithmPlan(
        title=title[:1].upper() + title[1:],
        problem_type=problem_type,
        inputs=inputs,
        outputs=outputs,
        steps=steps,
        notes=notes,
    )


def analyze_python_code(code: str) -> list[CodeSuggestion]:
    if not code.strip():
        raise ValueError("code cannot be empty.")

    suggestions: list[CodeSuggestion] = []
    lines = code.splitlines()
    tree = ast.parse(code)

    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "eval":
            suggestions.append(
                CodeSuggestion(
                    kind="security",
                    severity="high",
                    line=node.lineno,
                    message="`eval` bisa mengeksekusi input berbahaya.",
                    suggestion="Ganti `eval` dengan parser yang lebih aman atau pemetaan fungsi yang eksplisit.",
                )
            )
        elif isinstance(node, ast.ImportFrom) and any(alias.name == "*" for alias in node.names):
            suggestions.append(
                CodeSuggestion(
                    kind="maintainability",
                    severity="medium",
                    line=node.lineno,
                    message="Wildcard import membuat asal nama sulit dilacak.",
                    suggestion="Import nama yang memang dipakai saja.",
                )
            )
        elif isinstance(node, ast.ExceptHandler) and node.type is None:
            suggestions.append(
                CodeSuggestion(
                    kind="reliability",
                    severity="high",
                    line=node.lineno,
                    message="Bare `except` menangkap terlalu banyak error dan menyulitkan debugging.",
                    suggestion="Tangkap exception yang lebih spesifik, misalnya `except ValueError`.",
                )
            )
        elif isinstance(node, ast.ExceptHandler) and isinstance(node.type, ast.Name) and node.type.id == "Exception":
            suggestions.append(
                CodeSuggestion(
                    kind="reliability",
                    severity="medium",
                    line=node.lineno,
                    message="`except Exception` sering terlalu luas untuk code produksi.",
                    suggestion="Persempit jenis exception yang ditangani bila memungkinkan.",
                )
            )

    function_defs = [node for node in ast.walk(tree) if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))]
    if not function_defs:
        suggestions.append(
            CodeSuggestion(
                kind="structure",
                severity="low",
                line=1,
                message="Kode ini belum punya fungsi, jadi sulit diuji ulang atau dipakai ulang.",
                suggestion="Pindahkan logika utama ke fungsi bernama yang jelas.",
            )
        )

    for line_number, line in enumerate(lines, start=1):
        stripped = line.strip()
        if len(line) > 100:
            suggestions.append(
                CodeSuggestion(
                    kind="style",
                    severity="low",
                    line=line_number,
                    message="Baris terlalu panjang dan lebih sulit dibaca.",
                    suggestion="Pecah baris panjang menjadi beberapa bagian yang lebih jelas.",
                )
            )
        if stripped.startswith("print(") and "debug" in stripped.casefold():
            suggestions.append(
                CodeSuggestion(
                    kind="debugging",
                    severity="low",
                    line=line_number,
                    message="Debug print bisa tertinggal di code final.",
                    suggestion="Ganti dengan logging terstruktur atau hapus jika sudah tidak dibutuhkan.",
                )
            )

    if "__name__ == \"__main__\"" not in code and "__name__ == '__main__'" not in code and "argparse" in code:
        suggestions.append(
            CodeSuggestion(
                kind="entrypoint",
                severity="low",
                line=1,
                message="Script CLI ini belum terlihat punya main guard.",
                suggestion="Tambahkan `if __name__ == '__main__':` agar impor modul tetap aman.",
            )
        )

    suggestions.sort(key=lambda suggestion: ({"high": 0, "medium": 1, "low": 2}[suggestion.severity], suggestion.line))
    return suggestions


def analyze_python_file(path: str | Path) -> list[CodeSuggestion]:
    source_path = Path(path)
    return analyze_python_code(source_path.read_text(encoding="utf-8"))


def assess_ai_ethics(system_description: str) -> EthicsAssessment:
    normalized = _normalize_problem(system_description)
    if not normalized:
        raise ValueError("system_description cannot be empty.")

    flagged_categories: list[str] = []
    risks: list[str] = []
    mitigations: list[str] = []

    def add_flag(category: str, risk: str, mitigation: str) -> None:
        if category not in flagged_categories:
            flagged_categories.append(category)
        if risk not in risks:
            risks.append(risk)
        if mitigation not in mitigations:
            mitigations.append(mitigation)

    if any(keyword in normalized for keyword in ("name", "email", "phone", "location", "face", "voice", "identity", "personal data", "recording")):
        add_flag(
            "privacy",
            "Sistem ini tampak memproses data pribadi atau identitas pengguna.",
            "Minimalkan data yang dikumpulkan, minta consent, dan jelaskan retensi data secara jelas.",
        )
    if any(keyword in normalized for keyword in ("hiring", "recruit", "loan", "credit", "school", "police", "ranking", "score")):
        add_flag(
            "bias_and_fairness",
            "Keputusan berdampak tinggi berisiko mewarisi bias atau diskriminasi.",
            "Audit performa per kelompok, pakai human review, dan simpan alasan keputusan.",
        )
    if any(keyword in normalized for keyword in ("medical", "diagnosis", "health", "medicine", "autonomous", "car", "drone", "robot")):
        add_flag(
            "safety",
            "Kesalahan sistem bisa berdampak pada keselamatan atau kesehatan.",
            "Tambahkan batas penggunaan, validasi manusia, dan prosedur fallback yang aman.",
        )
    if any(keyword in normalized for keyword in ("predict", "recommend", "decision", "moderation", "ban", "approval", "score", "ranking")):
        add_flag(
            "transparency",
            "Pengguna perlu tahu bagaimana sistem dipakai untuk membuat rekomendasi atau keputusan.",
            "Sediakan penjelasan, confidence, dan alasan utama di balik keluaran sistem.",
        )
    if any(keyword in normalized for keyword in ("automatic", "fully automated", "tanpa manusia", "no human")):
        add_flag(
            "human_oversight",
            "Otomatisasi penuh bisa membuat kesalahan sulit dikoreksi sebelum berdampak.",
            "Tambahkan human-in-the-loop untuk keputusan penting atau kasus yang confidence-nya rendah.",
        )
    if any(keyword in normalized for keyword in ("password", "secret", "credential", "token", "code execution", "command")):
        add_flag(
            "security",
            "Sistem ini menyentuh area sensitif yang perlu perlindungan keamanan tambahan.",
            "Pisahkan secret, audit akses, validasi input, dan batasi aksi berbahaya.",
        )

    if "bias_and_fairness" in flagged_categories and ("privacy" in flagged_categories or "human_oversight" in flagged_categories):
        overall_risk = "high"
    elif len(flagged_categories) >= 4:
        overall_risk = "high"
    elif len(flagged_categories) >= 2:
        overall_risk = "medium"
    else:
        overall_risk = "low"

    if not risks:
        risks.append("Belum ada risiko besar yang langsung terdeteksi dari deskripsi singkat ini.")
        mitigations.append("Tetap lakukan review privasi, keamanan, dan human oversight sebelum deployment.")

    return EthicsAssessment(
        overall_risk=overall_risk,
        risks=risks,
        mitigations=mitigations,
        flagged_categories=flagged_categories,
    )


def format_algorithm_plan(plan: AlgorithmPlan) -> str:
    lines = [
        f"Title: {plan.title}",
        f"Problem type: {plan.problem_type}",
        f"Inputs: {', '.join(plan.inputs)}",
        f"Outputs: {', '.join(plan.outputs)}",
        "Steps:",
    ]
    lines.extend(f"  {index}. {step}" for index, step in enumerate(plan.steps, start=1))
    lines.append("Notes:")
    lines.extend(f"  - {note}" for note in plan.notes)
    return "\n".join(lines)


def format_code_suggestions(suggestions: list[CodeSuggestion]) -> str:
    if not suggestions:
        return "No major issues detected."
    lines = ["Programming Suggestions:"]
    for suggestion in suggestions:
        lines.append(
            f"  - line {suggestion.line} [{suggestion.severity}] {suggestion.kind}: "
            f"{suggestion.message} -> {suggestion.suggestion}"
        )
    return "\n".join(lines)


def format_ethics_assessment(assessment: EthicsAssessment) -> str:
    lines = [
        f"Overall risk: {assessment.overall_risk}",
        f"Flagged categories: {', '.join(assessment.flagged_categories) if assessment.flagged_categories else 'none'}",
        "Risks:",
    ]
    lines.extend(f"  - {risk}" for risk in assessment.risks)
    lines.append("Mitigations:")
    lines.extend(f"  - {mitigation}" for mitigation in assessment.mitigations)
    return "\n".join(lines)
