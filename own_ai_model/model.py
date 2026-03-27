from __future__ import annotations

from dataclasses import dataclass
import math
import random


@dataclass
class ModelGradients:
    w1: list[list[float]]
    b1: list[float]
    w2: list[float]
    b2: float = 0.0


@dataclass
class ModelVelocity:
    w1: list[list[float]]
    b1: list[float]
    w2: list[float]
    b2: float = 0.0


@dataclass(frozen=True)
class ParameterStats:
    parameter_count: int
    l1_norm: float
    l2_norm: float
    max_abs_value: float
    mean_abs_value: float

    def to_dict(self) -> dict[str, float | int]:
        return {
            "parameter_count": self.parameter_count,
            "l1_norm": self.l1_norm,
            "l2_norm": self.l2_norm,
            "max_abs_value": self.max_abs_value,
            "mean_abs_value": self.mean_abs_value,
        }


@dataclass(frozen=True)
class ForwardPassResult:
    input_vector: list[float]
    hidden_linear: list[float]
    hidden_activation: list[float]
    output_linear: float
    output_activation: float

    def to_dict(self) -> dict[str, object]:
        return {
            "input_vector": self.input_vector,
            "hidden_linear": self.hidden_linear,
            "hidden_activation": self.hidden_activation,
            "output_linear": self.output_linear,
            "output_activation": self.output_activation,
        }


@dataclass(frozen=True)
class LossBreakdown:
    label: int
    prediction: float
    loss_function: str
    data_loss: float
    regularization_loss: float
    total_loss: float

    def to_dict(self) -> dict[str, float | int | str]:
        return {
            "label": self.label,
            "prediction": self.prediction,
            "loss_function": self.loss_function,
            "data_loss": self.data_loss,
            "regularization_loss": self.regularization_loss,
            "total_loss": self.total_loss,
        }


class SimpleNeuralNetwork:
    def __init__(self, input_size: int, hidden_size: int, seed: int = 0, hidden_activation: str = "relu") -> None:
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.hidden_activation_name = hidden_activation

        rng = random.Random(seed)
        first_layer_scale = 1.0 / math.sqrt(max(1, input_size))
        second_layer_scale = 1.0 / math.sqrt(max(1, hidden_size))

        self.w1 = [
            [rng.uniform(-first_layer_scale, first_layer_scale) for _ in range(input_size)]
            for _ in range(hidden_size)
        ]
        self.b1 = [0.0 for _ in range(hidden_size)]
        self.w2 = [rng.uniform(-second_layer_scale, second_layer_scale) for _ in range(hidden_size)]
        self.b2 = 0.0

    @staticmethod
    def relu(value: float) -> float:
        return max(0.0, value)

    @staticmethod
    def _sigmoid(value: float) -> float:
        bounded = max(min(value, 60.0), -60.0)
        return 1.0 / (1.0 + math.exp(-bounded))

    @staticmethod
    def _safe_probability(probability: float) -> float:
        return min(max(probability, 1e-7), 1.0 - 1e-7)

    @classmethod
    def binary_cross_entropy(cls, probability: float, label: int) -> float:
        safe_probability = cls._safe_probability(probability)
        return -((label * math.log(safe_probability)) + ((1 - label) * math.log(1.0 - safe_probability)))

    @staticmethod
    def mean_squared_error(prediction: float, label: int) -> float:
        error = prediction - label
        return error * error

    @classmethod
    def _apply_hidden_activation(cls, value: float, activation_name: str) -> float:
        if activation_name == "relu":
            return cls.relu(value)
        if activation_name == "sigmoid":
            return cls._sigmoid(value)
        if activation_name == "tanh":
            return math.tanh(value)
        raise ValueError(f"Unsupported hidden activation: {activation_name}")

    @staticmethod
    def _hidden_activation_derivative(linear_value: float, activation_value: float, activation_name: str) -> float:
        if activation_name == "relu":
            return 1.0 if linear_value > 0.0 else 0.0
        if activation_name == "sigmoid":
            return activation_value * (1.0 - activation_value)
        if activation_name == "tanh":
            return 1.0 - (activation_value * activation_value)
        raise ValueError(f"Unsupported hidden activation: {activation_name}")

    @staticmethod
    def _output_activation_derivative(probability: float) -> float:
        return probability * (1.0 - probability)

    def blank_gradients(self) -> ModelGradients:
        return ModelGradients(
            w1=[[0.0 for _ in range(self.input_size)] for _ in range(self.hidden_size)],
            b1=[0.0 for _ in range(self.hidden_size)],
            w2=[0.0 for _ in range(self.hidden_size)],
        )

    def blank_velocity(self) -> ModelVelocity:
        return ModelVelocity(
            w1=[[0.0 for _ in range(self.input_size)] for _ in range(self.hidden_size)],
            b1=[0.0 for _ in range(self.hidden_size)],
            w2=[0.0 for _ in range(self.hidden_size)],
        )

    def parameter_stats(self) -> ParameterStats:
        values = [*self.b1, *self.w2, self.b2]
        for row in self.w1:
            values.extend(row)

        if not values:
            return ParameterStats(
                parameter_count=0,
                l1_norm=0.0,
                l2_norm=0.0,
                max_abs_value=0.0,
                mean_abs_value=0.0,
            )

        abs_values = [abs(value) for value in values]
        l1_norm = sum(abs_values)
        l2_norm = math.sqrt(sum(value * value for value in values))
        return ParameterStats(
            parameter_count=len(values),
            l1_norm=l1_norm,
            l2_norm=l2_norm,
            max_abs_value=max(abs_values),
            mean_abs_value=l1_norm / len(values),
        )

    def weight_l2_penalty(self) -> float:
        total = 0.0
        for row in self.w1:
            for value in row:
                total += value * value
        for value in self.w2:
            total += value * value
        return total

    @staticmethod
    def gradient_global_norm(gradients: ModelGradients) -> float:
        squared_sum = gradients.b2 * gradients.b2
        for value in gradients.b1:
            squared_sum += value * value
        for value in gradients.w2:
            squared_sum += value * value
        for row in gradients.w1:
            for value in row:
                squared_sum += value * value
        return math.sqrt(squared_sum)

    @classmethod
    def clip_gradients(cls, gradients: ModelGradients, clip_value: float | None) -> tuple[float, bool]:
        original_norm = cls.gradient_global_norm(gradients)
        if clip_value is None or original_norm <= clip_value or original_norm == 0.0:
            return original_norm, False

        scale = clip_value / original_norm
        gradients.b2 *= scale
        gradients.b1 = [value * scale for value in gradients.b1]
        gradients.w2 = [value * scale for value in gradients.w2]
        gradients.w1 = [[value * scale for value in row] for row in gradients.w1]
        return original_norm, True

    def forward(self, features: list[float]) -> tuple[list[float], float]:
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, received {len(features)}.")

        hidden: list[float] = []
        for node_index in range(self.hidden_size):
            weighted_sum = self.b1[node_index]
            for feature_index in range(self.input_size):
                weighted_sum += self.w1[node_index][feature_index] * features[feature_index]
            hidden.append(self._apply_hidden_activation(weighted_sum, self.hidden_activation_name))

        output_sum = self.b2
        for node_index in range(self.hidden_size):
            output_sum += self.w2[node_index] * hidden[node_index]

        return hidden, self._sigmoid(output_sum)

    def forward_pass(self, features: list[float]) -> ForwardPassResult:
        if len(features) != self.input_size:
            raise ValueError(f"Expected {self.input_size} features, received {len(features)}.")

        hidden_linear: list[float] = []
        hidden_activation: list[float] = []

        for node_index in range(self.hidden_size):
            weighted_sum = self.b1[node_index]
            for feature_index in range(self.input_size):
                weighted_sum += self.w1[node_index][feature_index] * features[feature_index]
            hidden_linear.append(weighted_sum)
            hidden_activation.append(self._apply_hidden_activation(weighted_sum, self.hidden_activation_name))

        output_linear = self.b2
        for node_index in range(self.hidden_size):
            output_linear += self.w2[node_index] * hidden_activation[node_index]

        return ForwardPassResult(
            input_vector=features[:],
            hidden_linear=hidden_linear,
            hidden_activation=hidden_activation,
            output_linear=output_linear,
            output_activation=self._sigmoid(output_linear),
        )

    def predict_probability(self, features: list[float]) -> float:
        return self.forward_pass(features).output_activation

    def predict_label(self, features: list[float]) -> int:
        return 1 if self.predict_probability(features) >= 0.5 else 0

    def loss_function(self, probability: float, label: int, l2_lambda: float = 0.0) -> LossBreakdown:
        data_loss = self.binary_cross_entropy(probability, label)
        regularization_loss = 0.5 * l2_lambda * self.weight_l2_penalty()
        return LossBreakdown(
            label=label,
            prediction=probability,
            loss_function="binary_cross_entropy",
            data_loss=data_loss,
            regularization_loss=regularization_loss,
            total_loss=data_loss + regularization_loss,
        )

    def loss_breakdown(
        self,
        probability: float,
        label: int,
        loss_function_name: str = "binary_cross_entropy",
        l2_lambda: float = 0.0,
    ) -> LossBreakdown:
        if loss_function_name == "binary_cross_entropy":
            data_loss = self.binary_cross_entropy(probability, label)
        elif loss_function_name == "mean_squared_error":
            data_loss = self.mean_squared_error(probability, label)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

        regularization_loss = 0.5 * l2_lambda * self.weight_l2_penalty()
        return LossBreakdown(
            label=label,
            prediction=probability,
            loss_function=loss_function_name,
            data_loss=data_loss,
            regularization_loss=regularization_loss,
            total_loss=data_loss + regularization_loss,
        )

    def loss_from_features(
        self,
        features: list[float],
        label: int,
        loss_function_name: str = "binary_cross_entropy",
        l2_lambda: float = 0.0,
    ) -> LossBreakdown:
        forward_result = self.forward_pass(features)
        return self.loss_breakdown(
            forward_result.output_activation,
            label,
            loss_function_name=loss_function_name,
            l2_lambda=l2_lambda,
        )

    @staticmethod
    def _scaled_error_signal(output_error: float, mistake_focus_power: float) -> float:
        if output_error == 0.0:
            return 0.0
        return math.copysign(abs(output_error) ** mistake_focus_power, output_error)

    def accumulate_gradients(
        self,
        features: list[float],
        label: int,
        gradients: ModelGradients,
        loss_function_name: str = "binary_cross_entropy",
        mistake_focus_power: float = 1.0,
    ) -> tuple[float, float]:
        forward_result = self.forward_pass(features)
        hidden = forward_result.hidden_activation
        hidden_linear = forward_result.hidden_linear
        probability = forward_result.output_activation
        raw_output_error = probability - label
        if loss_function_name == "binary_cross_entropy":
            base_output_delta = raw_output_error
        elif loss_function_name == "mean_squared_error":
            base_output_delta = 2.0 * raw_output_error * self._output_activation_derivative(probability)
        else:
            raise ValueError(f"Unsupported loss function: {loss_function_name}")

        output_delta = self._scaled_error_signal(base_output_delta, mistake_focus_power)

        for node_index in range(self.hidden_size):
            gradients.w2[node_index] += output_delta * hidden[node_index]

        gradients.b2 += output_delta

        for node_index in range(self.hidden_size):
            hidden_derivative = self._hidden_activation_derivative(
                hidden_linear[node_index],
                hidden[node_index],
                self.hidden_activation_name,
            )
            hidden_delta = hidden_derivative * self.w2[node_index] * output_delta
            gradients.b1[node_index] += hidden_delta
            for feature_index in range(self.input_size):
                gradients.w1[node_index][feature_index] += hidden_delta * features[feature_index]

        return self.loss_breakdown(probability, label, loss_function_name=loss_function_name).data_loss, abs(output_delta)

    def apply_gradients(
        self,
        gradients: ModelGradients,
        learning_rate: float,
        batch_size: int,
        l2_lambda: float = 0.0,
        velocity: ModelVelocity | None = None,
        momentum: float = 0.0,
        bias_learning_rate_multiplier: float = 1.0,
    ) -> None:
        scale = learning_rate / max(1, batch_size)
        bias_scale = scale * bias_learning_rate_multiplier

        for node_index in range(self.hidden_size):
            for feature_index in range(self.input_size):
                regularized_gradient = gradients.w1[node_index][feature_index] + (l2_lambda * self.w1[node_index][feature_index])
                if velocity is None:
                    self.w1[node_index][feature_index] -= scale * regularized_gradient
                else:
                    velocity.w1[node_index][feature_index] = (
                        (momentum * velocity.w1[node_index][feature_index]) - (scale * regularized_gradient)
                    )
                    self.w1[node_index][feature_index] += velocity.w1[node_index][feature_index]
            if velocity is None:
                self.b1[node_index] -= bias_scale * gradients.b1[node_index]
            else:
                velocity.b1[node_index] = (momentum * velocity.b1[node_index]) - (bias_scale * gradients.b1[node_index])
                self.b1[node_index] += velocity.b1[node_index]
            regularized_output_gradient = gradients.w2[node_index] + (l2_lambda * self.w2[node_index])
            if velocity is None:
                self.w2[node_index] -= scale * regularized_output_gradient
            else:
                velocity.w2[node_index] = (momentum * velocity.w2[node_index]) - (scale * regularized_output_gradient)
                self.w2[node_index] += velocity.w2[node_index]

        if velocity is None:
            self.b2 -= bias_scale * gradients.b2
        else:
            velocity.b2 = (momentum * velocity.b2) - (bias_scale * gradients.b2)
            self.b2 += velocity.b2

    def to_dict(self) -> dict[str, object]:
        return {
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "hidden_activation": self.hidden_activation_name,
            "w1": self.w1,
            "b1": self.b1,
            "w2": self.w2,
            "b2": self.b2,
        }

    def load_parameters(self, payload: dict[str, object]) -> None:
        self.hidden_activation_name = str(payload.get("hidden_activation", self.hidden_activation_name))
        self.w1 = [[float(value) for value in row] for row in payload["w1"]]
        self.b1 = [float(value) for value in payload["b1"]]
        self.w2 = [float(value) for value in payload["w2"]]
        self.b2 = float(payload["b2"])

    @classmethod
    def from_dict(cls, payload: dict[str, object]) -> "SimpleNeuralNetwork":
        input_size = int(payload["input_size"])
        hidden_size = int(payload["hidden_size"])
        hidden_activation = str(payload.get("hidden_activation", "tanh"))
        model = cls(input_size=input_size, hidden_size=hidden_size, seed=0, hidden_activation=hidden_activation)
        model.load_parameters(payload)
        return model
