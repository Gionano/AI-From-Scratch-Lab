from __future__ import annotations

from dataclasses import dataclass
import json
import re


STATUS_LABELS = {
    "available": "available now",
    "partial": "partial foundation",
    "planned": "planned next",
}

OVERVIEW_KEYWORDS = (
    "ai map",
    "ai stack",
    "peta ai",
    "semua ini",
    "all of this",
    "hierarchy",
    "roadmap",
)

COMPARISON_KEYWORDS = ("beda", "difference", "vs", "versus")
QUESTION_PREFIXES = (
    "apa itu ",
    "jelaskan ",
    "what is ",
    "explain ",
    "tell me about ",
)


@dataclass(frozen=True)
class AIConcept:
    key: str
    name: str
    parent_key: str | None
    description: str
    status: str
    aliases: tuple[str, ...] = ()

    def to_dict(self, children: list[dict[str, object]] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "key": self.key,
            "name": self.name,
            "parent_key": self.parent_key,
            "description": self.description,
            "status": self.status,
            "status_label": STATUS_LABELS[self.status],
            "aliases": list(self.aliases),
        }
        if children is not None:
            payload["children"] = children
        return payload


AI_CONCEPTS = (
    AIConcept(
        key="artificial_intelligence",
        name="Artificial Intelligence",
        parent_key=None,
        description="Payung besar untuk sistem yang bisa meniru kemampuan cerdas seperti memahami, memutuskan, dan membantu manusia.",
        status="partial",
        aliases=("ai",),
    ),
    AIConcept("reinforcement_learning", "Reinforcement Learning", "artificial_intelligence", "Belajar lewat trial and error memakai reward dan penalty.", "available", ("rl",)),
    AIConcept("speech_recognition", "Speech Recognition", "artificial_intelligence", "Mengubah suara manusia menjadi teks atau perintah.", "planned"),
    AIConcept("emergent_behavior", "Emergent Behavior", "artificial_intelligence", "Perilaku baru yang muncul ketika sistem cukup kompleks.", "planned"),
    AIConcept("machine_learning", "Machine Learning", "artificial_intelligence", "Cabang AI yang belajar dari data untuk membuat prediksi atau keputusan.", "available", ("ml",)),
    AIConcept("augmented_programming", "Augmented Programming", "artificial_intelligence", "Pemakaian AI untuk membantu coding, debugging, dan otomatisasi kerja developer.", "available"),
    AIConcept("algorithm_building", "Algorithm Building", "artificial_intelligence", "Merancang prosedur langkah demi langkah untuk menyelesaikan masalah.", "available"),
    AIConcept("ai_ethics", "AI Ethics", "artificial_intelligence", "Prinsip keamanan, keadilan, privasi, dan tanggung jawab dalam pemakaian AI.", "available"),
    AIConcept("unsupervised_learning", "Unsupervised Learning", "machine_learning", "Belajar dari data tanpa label, misalnya clustering dan reduksi dimensi.", "planned"),
    AIConcept("supervised_learning", "Supervised Learning", "machine_learning", "Belajar dari data berlabel untuk klasifikasi atau regresi.", "available"),
    AIConcept("feature_engineering", "Feature Engineering", "machine_learning", "Mengubah input mentah menjadi fitur yang lebih berguna untuk model.", "available"),
    AIConcept("k_nearest_neighbours", "K-Nearest Neighbours", "machine_learning", "Prediksi berdasarkan tetangga data yang paling dekat.", "available", ("knn", "k nearest neighbours")),
    AIConcept("logistic_regression", "Logistic Regression", "machine_learning", "Model klasifikasi probabilistik yang sederhana dan kuat sebagai baseline.", "available"),
    AIConcept("linear_regression", "Linear Regression", "machine_learning", "Model garis lurus untuk memprediksi nilai numerik.", "planned"),
    AIConcept("pca", "PCA", "machine_learning", "Principal Component Analysis untuk meringkas dimensi data.", "planned"),
    AIConcept("hypothesis_testing", "Hypothesis Testing", "machine_learning", "Menguji dugaan statistik tentang pola di data.", "planned"),
    AIConcept("support_vector_machines", "Support Vector Machines", "machine_learning", "Model yang mencari batas keputusan terbaik antar kelas.", "planned", ("svm",)),
    AIConcept("decision_trees", "Decision Trees", "machine_learning", "Model berbasis aturan bercabang untuk klasifikasi atau regresi.", "available"),
    AIConcept("k_means", "K Means", "machine_learning", "Algoritma clustering yang mengelompokkan data ke pusat cluster terdekat.", "planned", ("k-means",)),
    AIConcept("neural_networks", "Neural Networks", "machine_learning", "Model berlapis yang belajar representasi dari data lewat bobot dan bias.", "available", ("nn",)),
    AIConcept("perceptron", "Perceptron", "neural_networks", "Neuron buatan paling dasar untuk klasifikasi sederhana.", "partial"),
    AIConcept("backpropagation", "Backpropagation", "neural_networks", "Cara menghitung gradien agar model belajar dari kesalahan.", "available", ("backprop",)),
    AIConcept("feed_forward", "Feed Forward", "neural_networks", "Aliran data maju dari input ke output tanpa loop waktu.", "available", ("feedforward",)),
    AIConcept("recurrent_neural_networks", "RNN", "neural_networks", "Neural network yang menyimpan konteks urutan dari langkah sebelumnya.", "planned", ("rnn", "recurrent neural networks")),
    AIConcept("hopfield_network", "Hopfield Network", "neural_networks", "Jaringan rekuren klasik untuk memori asosiasi.", "planned"),
    AIConcept("liquid_state_machines", "Liquid State Machines", "neural_networks", "Model komputasi reservoir untuk sinyal temporal.", "planned"),
    AIConcept("self_organising_maps", "Self Organising Maps", "neural_networks", "Pemetaan data ke grid yang menjaga kemiripan topologi.", "planned", ("som", "self organizing maps", "self organising maps")),
    AIConcept("boltzmann_machine", "Boltzmann Machine", "neural_networks", "Model probabilistik berbasis energi untuk representasi dan sampling.", "planned"),
    AIConcept("deep_learning", "Deep Learning", "neural_networks", "Neural network yang lebih dalam dan kuat untuk pola kompleks seperti gambar, teks, dan audio.", "partial", ("dl",)),
    AIConcept("deep_reinforcement_learning", "Deep Reinforcement Learning", "deep_learning", "Gabungan deep learning dan reinforcement learning.", "planned"),
    AIConcept("transformers", "Transformers", "deep_learning", "Arsitektur modern yang kuat untuk bahasa, visi, dan multimodal.", "planned"),
    AIConcept("epochs", "Epochs", "deep_learning", "Jumlah putaran penuh ketika model melihat seluruh data training.", "available", ("epoch",)),
    AIConcept("cnn", "CNN", "deep_learning", "Convolutional Neural Network untuk pola spasial seperti gambar.", "planned", ("convolutional neural networks",)),
    AIConcept("diffusion_models", "Diffusion Models", "deep_learning", "Model generatif yang belajar mengubah noise menjadi sampel baru.", "planned"),
    AIConcept("autoencoders", "Auto Encoders", "deep_learning", "Model encoder-decoder untuk kompresi dan representasi laten.", "planned", ("autoencoders", "auto encoders")),
    AIConcept("deep_belief_network", "Deep Belief Network", "deep_learning", "Stack model probabilistik generatif dari era awal deep learning.", "planned"),
    AIConcept("generative_ai", "Generative AI", "deep_learning", "Bagian deep learning yang fokus menghasilkan teks, gambar, audio, atau kode baru.", "partial", ("genai", "gen ai")),
    AIConcept("n_shot_learning", "N-Shot Learning", "generative_ai", "Kemampuan menyesuaikan diri dari sejumlah kecil contoh.", "planned"),
    AIConcept("one_shot_learning", "One-Shot Learning", "generative_ai", "Belajar dari satu contoh saja.", "planned", ("osl",)),
    AIConcept("zero_shot_learning", "Zero-Shot Learning", "generative_ai", "Menyelesaikan tugas baru tanpa contoh khusus di training.", "planned", ("zsl",)),
    AIConcept("lora", "LoRA", "generative_ai", "Fine-tuning hemat parameter untuk model besar atau kecil.", "planned"),
    AIConcept("agents", "Agents", "generative_ai", "Sistem yang tidak hanya menjawab, tetapi juga merencanakan dan menjalankan aksi.", "planned"),
    AIConcept("gans", "Generative Adversarial Networks", "generative_ai", "Dua model yang saling menantang untuk menghasilkan data baru.", "planned", ("gan", "gans")),
    AIConcept("ensemble_models", "Ensemble Model", "generative_ai", "Menggabungkan beberapa model untuk hasil yang lebih kuat atau stabil.", "planned", ("ensemble",)),
    AIConcept("biggan", "BigGAN", "generative_ai", "Varian GAN skala besar untuk generasi gambar.", "planned"),
    AIConcept("functional_models", "Functional Models", "generative_ai", "Model yang dirancang untuk peran atau fungsi tertentu dalam sistem yang lebih besar.", "planned"),
    AIConcept("large_language_models", "Large Language Model (LLM)", "generative_ai", "Model bahasa besar untuk memahami dan menghasilkan teks secara fleksibel.", "planned", ("llm", "large language model")),
    AIConcept("gpt", "GPT", "generative_ai", "Keluarga model transformer autoregresif untuk bahasa dan coding.", "planned"),
    AIConcept("bert", "BERT", "generative_ai", "Model transformer bidirectional yang kuat untuk pemahaman bahasa.", "planned"),
    AIConcept("small_language_models", "Small Language Models (SLM)", "generative_ai", "Model bahasa yang lebih kecil, ringan, hemat sumber daya, dan cocok untuk device lokal.", "planned", ("slm", "small language model")),
)

AI_CONCEPT_INDEX = {concept.key: concept for concept in AI_CONCEPTS}
AI_CHILDREN_INDEX: dict[str | None, list[AIConcept]] = {}
for concept in AI_CONCEPTS:
    AI_CHILDREN_INDEX.setdefault(concept.parent_key, []).append(concept)


def _normalize(text: str) -> str:
    lowered = text.casefold()
    cleaned = re.sub(r"[^a-z0-9+ ]+", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def _search_terms(concept: AIConcept) -> list[str]:
    terms = {
        _normalize(concept.key.replace("_", " ")),
        _normalize(concept.name),
    }
    for alias in concept.aliases:
        terms.add(_normalize(alias))
    return sorted(term for term in terms if term)


def list_ai_concepts() -> list[AIConcept]:
    return list(AI_CONCEPTS)


def get_ai_concept(key: str) -> AIConcept:
    return AI_CONCEPT_INDEX[key]


def get_ai_children(parent_key: str | None) -> list[AIConcept]:
    return list(AI_CHILDREN_INDEX.get(parent_key, []))


def get_ai_concept_path(key: str) -> list[AIConcept]:
    path: list[AIConcept] = []
    current = get_ai_concept(key)
    while True:
        path.append(current)
        if current.parent_key is None:
            break
        current = get_ai_concept(current.parent_key)
    path.reverse()
    return path


def match_ai_concept(query: str) -> AIConcept | None:
    normalized_query = _normalize(query)
    if not normalized_query:
        return None

    matches: list[tuple[int, AIConcept]] = []
    padded_query = f" {normalized_query} "
    for concept in AI_CONCEPTS:
        for term in _search_terms(concept):
            if normalized_query == term or f" {term} " in padded_query:
                matches.append((len(term), concept))
    if not matches:
        return None
    matches.sort(key=lambda item: item[0], reverse=True)
    return matches[0][1]


def _find_all_concepts(query: str) -> list[AIConcept]:
    normalized_query = _normalize(query)
    padded_query = f" {normalized_query} "
    found: list[AIConcept] = []
    seen: set[str] = set()
    for concept in AI_CONCEPTS:
        for term in _search_terms(concept):
            if normalized_query == term or f" {term} " in padded_query:
                if concept.key not in seen:
                    seen.add(concept.key)
                    found.append(concept)
                break
    found.sort(key=lambda concept: len(concept.name), reverse=True)
    return found


def _format_status(status: str) -> str:
    return STATUS_LABELS.get(status, status)


def _tree_lines(parent_key: str | None, depth: int) -> list[str]:
    lines: list[str] = []
    for concept in get_ai_children(parent_key):
        indent = "  " * depth
        lines.append(f"{indent}- {concept.name} [{_format_status(concept.status)}]")
        lines.extend(_tree_lines(concept.key, depth + 1))
    return lines


def format_ai_stack_tree() -> str:
    lines = ["AI Stack Map", * _tree_lines(None, 0)]
    lines.append("")
    lines.append("Current project focus: reinforcement learning, algorithm planning, augmented programming, AI ethics, supervised learning, logistic regression, k-nearest neighbours, decision trees, feed forward, backpropagation, explicit loss, and a local chatbot foundation.")
    return "\n".join(lines)


def build_ai_stack_payload() -> list[dict[str, object]]:
    def build_node(concept: AIConcept) -> dict[str, object]:
        return concept.to_dict(children=[build_node(child) for child in get_ai_children(concept.key)])

    return [build_node(concept) for concept in get_ai_children(None)]


def format_ai_concept_details(key_or_query: str) -> str:
    concept = AI_CONCEPT_INDEX.get(key_or_query) or match_ai_concept(key_or_query)
    if concept is None:
        raise KeyError(f"Unknown AI concept: {key_or_query}")

    path = " -> ".join(node.name for node in get_ai_concept_path(concept.key))
    children = get_ai_children(concept.key)
    child_names = ", ".join(child.name for child in children[:8]) if children else "Tidak ada turunan langsung."
    return "\n".join(
        [
            f"{concept.name} [{_format_status(concept.status)}]",
            f"Path: {path}",
            f"Description: {concept.description}",
            f"Children: {child_names}",
        ]
    )


def _extract_topic(normalized_message: str) -> str:
    for prefix in QUESTION_PREFIXES:
        if normalized_message.startswith(prefix):
            return normalized_message[len(prefix):].strip()
    return normalized_message


def _is_ancestor(ancestor_key: str, descendant_key: str) -> bool:
    current = get_ai_concept(descendant_key)
    while current.parent_key is not None:
        if current.parent_key == ancestor_key:
            return True
        current = get_ai_concept(current.parent_key)
    return False


def _format_comparison(left: AIConcept, right: AIConcept) -> str:
    if _is_ancestor(left.key, right.key):
        relation = f"{right.name} adalah bagian yang lebih spesifik di dalam {left.name}."
    elif _is_ancestor(right.key, left.key):
        relation = f"{left.name} adalah bagian yang lebih spesifik di dalam {right.name}."
    elif left.parent_key and left.parent_key == right.parent_key:
        parent = get_ai_concept(left.parent_key)
        relation = f"{left.name} dan {right.name} sama-sama berada di bawah {parent.name}, tetapi fokusnya berbeda."
    else:
        relation = f"{left.name} dan {right.name} berada di area yang berbeda dalam peta AI ini."

    return (
        f"{relation} "
        f"{left.name}: {left.description} "
        f"{right.name}: {right.description}"
    )


def answer_ai_stack_question(message: str) -> str | None:
    normalized_message = _normalize(message)
    if not normalized_message:
        return None

    if any(keyword in normalized_message for keyword in OVERVIEW_KEYWORDS):
        return (
            "Urutannya paling umum ke paling spesifik adalah: "
            "Artificial Intelligence -> Machine Learning -> Neural Networks -> Deep Learning -> Generative AI -> Small Language Models (SLM). "
            "Kalau mau lihat daftar lengkapnya di project ini, jalankan `show_ai_stack.py`."
        )

    mentioned = _find_all_concepts(normalized_message)
    if len(mentioned) >= 2 and any(keyword in normalized_message for keyword in COMPARISON_KEYWORDS):
        return _format_comparison(mentioned[0], mentioned[1])

    topic = _extract_topic(normalized_message)
    concept = match_ai_concept(topic) or match_ai_concept(normalized_message)
    if concept is None:
        return None

    children = get_ai_children(concept.key)
    child_text = ""
    if children:
        preview = ", ".join(child.name for child in children[:5])
        child_text = f" Contoh topik di bawahnya: {preview}."

    return (
        f"{concept.name} adalah {_format_status(concept.status)} di project ini. "
        f"{concept.description}{child_text}"
    )


def ai_stack_json(indent: int = 2) -> str:
    return json.dumps(build_ai_stack_payload(), indent=indent)
