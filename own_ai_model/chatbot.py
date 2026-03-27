from __future__ import annotations

from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
from pathlib import Path
import random
import re

from .ai_stack import answer_ai_stack_question
from .runtime import RuntimeInfo

TOKEN_PATTERN = re.compile(r"[a-z0-9']+")
NAME_PATTERN = re.compile(r"\b(?:nama saya|panggil saya|my name is|call me)\s+([^,.!?]+)", re.IGNORECASE)
NOTE_PATTERN = re.compile(r"\b(?:ingat bahwa|tolong ingat bahwa|catat bahwa|remember that)\s+(.+)", re.IGNORECASE)
KEYWORD_HINTS = {
    "coding_help": {"error", "bug", "debug", "kode", "coding", "program"},
    "learning_help": {"belajar", "materi", "python", "machine", "learning"},
    "feeling_down": {"sedih", "stres", "capek", "kecewa", "buruk"},
    "feeling_good": {"senang", "bahagia", "semangat", "bangga"},
    "planning": {"rencana", "roadmap", "mulai", "langkah", "target"},
}


def normalize_text(text: str) -> str:
    lowered = text.casefold()
    collapsed = re.sub(r"\s+", " ", lowered)
    return collapsed.strip()


def tokenize(text: str) -> list[str]:
    return TOKEN_PATTERN.findall(normalize_text(text))


@dataclass(frozen=True)
class ChatbotExample:
    text: str
    intent: str


@dataclass(frozen=True)
class ChatbotTrainingConfig:
    data_path: str
    model_path: str
    epochs: int
    learning_rate: float
    report_every: int
    min_word_frequency: int = 1
    confidence_threshold: float = 0.34
    seed: int = 7

    def __post_init__(self) -> None:
        if not self.data_path.strip():
            raise ValueError("data_path cannot be empty.")
        if not self.model_path.strip():
            raise ValueError("model_path cannot be empty.")
        if self.epochs < 1:
            raise ValueError("epochs must be at least 1.")
        if self.learning_rate <= 0:
            raise ValueError("learning_rate must be greater than 0.")
        if self.report_every < 1:
            raise ValueError("report_every must be at least 1.")
        if self.min_word_frequency < 1:
            raise ValueError("min_word_frequency must be at least 1.")
        if not 0.0 < self.confidence_threshold < 1.0:
            raise ValueError("confidence_threshold must be between 0 and 1.")

    def to_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True)
class ChatbotTrainingResult:
    loss: float
    accuracy: float
    history: list[dict[str, float | int]]

    def to_dict(self) -> dict[str, object]:
        return {
            "loss": self.loss,
            "accuracy": self.accuracy,
            "history": self.history,
        }


@dataclass(frozen=True)
class ChatbotReply:
    intent: str
    confidence: float
    reply: str
    probabilities: dict[str, float]


@dataclass
class ChatSessionMemory:
    user_name: str | None = None
    notes: list[str] = field(default_factory=list)


class IntentClassifier:
    def __init__(self, vocabulary: list[str], labels: list[str], seed: int = 0) -> None:
        if not vocabulary:
            raise ValueError("vocabulary cannot be empty.")
        if not labels:
            raise ValueError("labels cannot be empty.")

        self.vocabulary = vocabulary
        self.labels = labels
        self.vocabulary_index = {token: index for index, token in enumerate(vocabulary)}
        self.label_index = {label: index for index, label in enumerate(labels)}

        rng = random.Random(seed)
        scale = 1.0 / math.sqrt(max(1, len(vocabulary)))
        self.weights = [
            [rng.uniform(-scale, scale) for _ in range(len(vocabulary))]
            for _ in range(len(labels))
        ]
        self.bias = [0.0 for _ in range(len(labels))]

    def vectorize_text(self, text: str) -> list[float]:
        tokens = tokenize(text)
        features = [0.0 for _ in range(len(self.vocabulary))]
        if not tokens:
            return features

        for token in tokens:
            if token in self.vocabulary_index:
                features[self.vocabulary_index[token]] += 1.0

        token_count = float(len(tokens))
        return [value / token_count for value in features]

    def predict_probabilities_from_features(self, features: list[float]) -> list[float]:
        logits = []
        for label_index in range(len(self.labels)):
            total = self.bias[label_index]
            for feature_index, feature_value in enumerate(features):
                if feature_value:
                    total += self.weights[label_index][feature_index] * feature_value
            logits.append(total)

        max_logit = max(logits)
        shifted = [math.exp(logit - max_logit) for logit in logits]
        total = sum(shifted)
        return [value / total for value in shifted]

    def predict(self, text: str) -> tuple[str, float, dict[str, float]]:
        features = self.vectorize_text(text)
        probabilities = self.predict_probabilities_from_features(features)
        best_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        return (
            self.labels[best_index],
            probabilities[best_index],
            {label: probabilities[index] for index, label in enumerate(self.labels)},
        )

    def load_parameters(self, payload: dict[str, object]) -> None:
        self.weights = [[float(value) for value in row] for row in payload["weights"]]
        self.bias = [float(value) for value in payload["bias"]]

    def to_dict(self) -> dict[str, object]:
        return {
            "vocabulary": self.vocabulary,
            "labels": self.labels,
            "weights": self.weights,
            "bias": self.bias,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "IntentClassifier":
        classifier = cls(
            vocabulary=[str(value) for value in payload["vocabulary"]],
            labels=[str(value) for value in payload["labels"]],
            seed=0,
        )
        classifier.load_parameters(payload)
        return classifier


def build_vocabulary(examples: list[ChatbotExample], min_word_frequency: int) -> list[str]:
    counts: dict[str, int] = {}
    for example in examples:
        for token in tokenize(example.text):
            counts[token] = counts.get(token, 0) + 1

    vocabulary = sorted(token for token, count in counts.items() if count >= min_word_frequency)
    if not vocabulary:
        raise ValueError("Vocabulary is empty. Add more training phrases or lower min_word_frequency.")
    return vocabulary


def load_chatbot_config(path: str | Path) -> ChatbotTrainingConfig:
    config_path = Path(path)
    raw_config = json.loads(config_path.read_text(encoding="utf-8"))
    return ChatbotTrainingConfig(**raw_config)


def load_intent_dataset(path: str | Path) -> tuple[list[ChatbotExample], dict[str, list[str]], list[str]]:
    dataset_path = Path(path)
    payload = json.loads(dataset_path.read_text(encoding="utf-8"))

    intents = payload.get("intents", [])
    if not intents:
        raise ValueError("Intent dataset must contain at least one intent.")

    examples: list[ChatbotExample] = []
    responses_by_intent: dict[str, list[str]] = {}

    for intent_payload in intents:
        tag = str(intent_payload["tag"]).strip()
        patterns = [str(value).strip() for value in intent_payload.get("patterns", []) if str(value).strip()]
        responses = [str(value).strip() for value in intent_payload.get("responses", []) if str(value).strip()]

        if not tag:
            raise ValueError("Intent tag cannot be empty.")
        if not patterns:
            raise ValueError(f"Intent '{tag}' must contain at least one pattern.")
        if not responses:
            raise ValueError(f"Intent '{tag}' must contain at least one response.")

        responses_by_intent[tag] = responses
        for pattern in patterns:
            examples.append(ChatbotExample(text=pattern, intent=tag))

    fallback_responses = [
        str(value).strip()
        for value in payload.get("fallback_responses", [])
        if str(value).strip()
    ]
    if not fallback_responses:
        fallback_responses = [
            "Aku belum terlalu yakin dengan maksudmu, {name}. Coba jelaskan sedikit lebih spesifik.",
            "Aku masih belajar memahami kalimat itu. Coba tulis ulang dengan cara yang lebih sederhana, {name}.",
        ]

    return examples, responses_by_intent, fallback_responses


def train_intent_classifier(
    examples: list[ChatbotExample],
    config: ChatbotTrainingConfig,
    verbose: bool = True,
) -> tuple[IntentClassifier, ChatbotTrainingResult]:
    vocabulary = build_vocabulary(examples, config.min_word_frequency)
    labels = sorted({example.intent for example in examples})
    classifier = IntentClassifier(vocabulary=vocabulary, labels=labels, seed=config.seed)

    feature_rows = [classifier.vectorize_text(example.text) for example in examples]
    label_indexes = [classifier.label_index[example.intent] for example in examples]
    history: list[dict[str, float | int]] = []

    for epoch in range(1, config.epochs + 1):
        total_loss = 0.0
        correct_predictions = 0
        gradient_w = [[0.0 for _ in range(len(vocabulary))] for _ in range(len(labels))]
        gradient_b = [0.0 for _ in range(len(labels))]

        for features, label_index in zip(feature_rows, label_indexes, strict=True):
            probabilities = classifier.predict_probabilities_from_features(features)
            predicted_index = max(range(len(probabilities)), key=probabilities.__getitem__)
            if predicted_index == label_index:
                correct_predictions += 1

            total_loss += -math.log(max(probabilities[label_index], 1e-9))

            for class_index in range(len(labels)):
                target = 1.0 if class_index == label_index else 0.0
                error = probabilities[class_index] - target
                gradient_b[class_index] += error
                for feature_index, feature_value in enumerate(features):
                    if feature_value:
                        gradient_w[class_index][feature_index] += error * feature_value

        scale = config.learning_rate / len(examples)
        for class_index in range(len(labels)):
            classifier.bias[class_index] -= scale * gradient_b[class_index]
            for feature_index in range(len(vocabulary)):
                classifier.weights[class_index][feature_index] -= scale * gradient_w[class_index][feature_index]

        epoch_record = {
            "epoch": epoch,
            "loss": total_loss / len(examples),
            "accuracy": correct_predictions / len(examples),
        }
        history.append(epoch_record)

        should_report = (
            verbose
            and (
                epoch == 1
                or epoch == config.epochs
                or epoch % config.report_every == 0
            )
        )
        if should_report:
            print(
                f"Epoch {epoch:>3}/{config.epochs} | "
                f"loss {epoch_record['loss']:.4f} | "
                f"accuracy {epoch_record['accuracy']:.2%}"
            )

    last_record = history[-1]
    result = ChatbotTrainingResult(
        loss=float(last_record["loss"]),
        accuracy=float(last_record["accuracy"]),
        history=history,
    )
    return classifier, result


def save_chatbot_model(
    path: str | Path,
    classifier: IntentClassifier,
    config: ChatbotTrainingConfig,
    responses_by_intent: dict[str, list[str]],
    fallback_responses: list[str],
    training_result: ChatbotTrainingResult,
    runtime_info: RuntimeInfo | None = None,
) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "config": config.to_dict(),
        "classifier": classifier.to_dict(),
        "responses_by_intent": responses_by_intent,
        "fallback_responses": fallback_responses,
        "training_result": training_result.to_dict(),
        "runtime": runtime_info.to_dict() if runtime_info is not None else {},
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def load_chatbot_model(
    path: str | Path,
) -> tuple[IntentClassifier, dict[str, list[str]], list[str], dict[str, object]]:
    model_path = Path(path)
    payload = json.loads(model_path.read_text(encoding="utf-8"))
    classifier = IntentClassifier.from_dict(payload["classifier"])
    responses_by_intent = {
        str(intent): [str(value) for value in responses]
        for intent, responses in payload["responses_by_intent"].items()
    }
    fallback_responses = [str(value) for value in payload.get("fallback_responses", [])]
    return classifier, responses_by_intent, fallback_responses, payload


def _format_with_name(template: str, memory: ChatSessionMemory) -> str:
    return template.format(name=memory.user_name or "teman")


def _choose_response(intent: str, normalized_text: str, responses: dict[str, list[str]], memory: ChatSessionMemory) -> str:
    options = responses[intent]
    digest = hashlib.sha256(f"{intent}:{normalized_text}".encode("utf-8")).digest()
    choice_index = digest[0] % len(options)
    return _format_with_name(options[choice_index], memory)


def _extract_name(message: str) -> str | None:
    match = NAME_PATTERN.search(message)
    if not match:
        return None

    raw_name = match.group(1).strip()
    collected: list[str] = []
    for token in raw_name.split():
        cleaned = token.strip(".,!?")
        if normalize_text(cleaned) in {"dan", "tapi", "karena", "yang"}:
            break
        if cleaned:
            collected.append(cleaned)
        if len(collected) >= 3:
            break

    if not collected:
        return None

    return " ".join(part.capitalize() for part in collected)


def _extract_note(message: str) -> str | None:
    match = NOTE_PATTERN.search(message)
    if not match:
        return None

    note = match.group(1).strip()
    if len(note) < 4:
        return None
    return note.rstrip(".!?")


def _handle_memory(message: str, memory: ChatSessionMemory) -> str | None:
    normalized = normalize_text(message)
    extracted_name = _extract_name(message)
    if extracted_name:
        memory.user_name = extracted_name
        return f"Siap, aku ingat. Aku akan memanggil kamu {extracted_name}."

    extracted_note = _extract_note(message)
    if extracted_note:
        if extracted_note not in memory.notes:
            memory.notes.append(extracted_note)
        return f"Baik, aku simpan catatan ini: {extracted_note}."

    if normalized in {
        "siapa nama saya",
        "kamu ingat nama saya",
        "what is my name",
    }:
        if memory.user_name:
            return f"Nama yang aku ingat adalah {memory.user_name}."
        return "Aku belum tahu nama kamu. Coba tulis: nama saya Budi."

    if normalized in {
        "apa yang kamu ingat tentang saya",
        "apa yang kamu ingat",
        "what do you remember about me",
    }:
        remembered: list[str] = []
        if memory.user_name:
            remembered.append(f"nama kamu {memory.user_name}")
        remembered.extend(memory.notes[-3:])
        if remembered:
            return "Yang aku ingat saat ini: " + "; ".join(remembered) + "."
        return "Saat ini aku belum menyimpan apa pun tentang kamu."

    if normalized in {
        "hapus ingatan",
        "lupakan saya",
        "clear memory",
    }:
        memory.user_name = None
        memory.notes.clear()
        return "Oke, memori sesi sudah aku bersihkan."

    return None


def _match_keyword_intent(message: str) -> str | None:
    tokens = set(tokenize(message))
    for intent, keywords in KEYWORD_HINTS.items():
        if tokens & keywords:
            return intent
    return None


def generate_chatbot_reply(
    message: str,
    classifier: IntentClassifier,
    responses_by_intent: dict[str, list[str]],
    fallback_responses: list[str],
    memory: ChatSessionMemory | None = None,
    confidence_threshold: float = 0.34,
) -> ChatbotReply:
    if not message.strip():
        return ChatbotReply(
            intent="empty_input",
            confidence=1.0,
            reply="Coba tulis kalimatmu dulu, nanti aku jawab.",
            probabilities={},
        )

    active_memory = memory if memory is not None else ChatSessionMemory()
    memory_response = _handle_memory(message, active_memory)
    if memory_response is not None:
        return ChatbotReply(
            intent="memory",
            confidence=1.0,
            reply=memory_response,
            probabilities={},
        )

    ai_stack_response = answer_ai_stack_question(message)
    if ai_stack_response is not None:
        return ChatbotReply(
            intent="ai_knowledge",
            confidence=1.0,
            reply=ai_stack_response,
            probabilities={},
        )

    predicted_intent, confidence, probabilities = classifier.predict(message)
    normalized = normalize_text(message)

    if confidence < confidence_threshold:
        hinted_intent = _match_keyword_intent(message)
        if hinted_intent and hinted_intent in responses_by_intent:
            reply = _choose_response(hinted_intent, normalized, responses_by_intent, active_memory)
            return ChatbotReply(
                intent=hinted_intent,
                confidence=max(confidence, 0.75),
                reply=reply,
                probabilities=probabilities,
            )

        digest = hashlib.sha256(normalized.encode("utf-8")).digest()
        reply = _format_with_name(fallback_responses[digest[0] % len(fallback_responses)], active_memory)
        return ChatbotReply(
            intent="fallback",
            confidence=confidence,
            reply=reply,
            probabilities=probabilities,
        )

    reply = _choose_response(predicted_intent, normalized, responses_by_intent, active_memory)
    return ChatbotReply(
        intent=predicted_intent,
        confidence=confidence,
        reply=reply,
        probabilities=probabilities,
    )
