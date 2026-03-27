#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <vector>

struct Example {
    std::vector<double> features;
    int label;
};

struct Config {
    int seed = 7;
    int train_samples = 900;
    int test_samples = 300;
    double coordinate_min = -1.0;
    double coordinate_max = 1.0;
    double circle_radius = 0.78;
    double center_x = 0.2;
    double center_y = -0.1;
    int hidden_size = 10;
    int epochs = 180;
    double learning_rate = 0.08;
    std::string hidden_activation = "relu";
    std::string loss_function = "binary_cross_entropy";
};

struct NeuralNetwork {
    int input_size = 2;
    int hidden_size = 10;
    std::string hidden_activation = "relu";
    std::vector<std::vector<double>> w1;
    std::vector<double> b1;
    std::vector<double> w2;
    double b2 = 0.0;
};

struct ForwardPass {
    std::vector<double> hidden_z;
    std::vector<double> hidden_a;
    double output_z;
    double output_y;
};

double relu(double value) {
    return std::max(0.0, value);
}

double sigmoid(double value) {
    const double bounded = std::max(-60.0, std::min(60.0, value));
    return 1.0 / (1.0 + std::exp(-bounded));
}

double activation_forward(double value, const std::string& name) {
    if (name == "relu") {
        return relu(value);
    }
    if (name == "sigmoid") {
        return sigmoid(value);
    }
    throw std::runtime_error("Unsupported activation.");
}

double activation_derivative(double linear_value, double activation_value, const std::string& name) {
    if (name == "relu") {
        return linear_value > 0.0 ? 1.0 : 0.0;
    }
    if (name == "sigmoid") {
        return activation_value * (1.0 - activation_value);
    }
    throw std::runtime_error("Unsupported activation.");
}

double data_loss(double probability, int label, const std::string& loss_name) {
    const double safe_probability = std::max(1e-7, std::min(1.0 - 1e-7, probability));
    if (loss_name == "binary_cross_entropy") {
        return -((label * std::log(safe_probability)) + ((1 - label) * std::log(1.0 - safe_probability)));
    }
    if (loss_name == "mean_squared_error") {
        const double error = probability - label;
        return error * error;
    }
    throw std::runtime_error("Unsupported loss function.");
}

double output_delta(double probability, int label, const std::string& loss_name) {
    if (loss_name == "binary_cross_entropy") {
        return probability - label;
    }
    if (loss_name == "mean_squared_error") {
        return 2.0 * (probability - label) * probability * (1.0 - probability);
    }
    throw std::runtime_error("Unsupported loss function.");
}

NeuralNetwork init_model(const Config& config) {
    NeuralNetwork model;
    model.hidden_size = config.hidden_size;
    model.hidden_activation = config.hidden_activation;
    model.w1.assign(config.hidden_size, std::vector<double>(2, 0.0));
    model.b1.assign(config.hidden_size, 0.0);
    model.w2.assign(config.hidden_size, 0.0);

    std::mt19937 rng(config.seed);
    const double first_scale = 1.0 / std::sqrt(2.0);
    const double second_scale = 1.0 / std::sqrt(static_cast<double>(config.hidden_size));
    std::uniform_real_distribution<double> first_dist(-first_scale, first_scale);
    std::uniform_real_distribution<double> second_dist(-second_scale, second_scale);

    for (int i = 0; i < config.hidden_size; ++i) {
        for (int j = 0; j < 2; ++j) {
            model.w1[i][j] = first_dist(rng);
        }
        model.w2[i] = second_dist(rng);
    }
    return model;
}

int classify_point(double x, double y, const Config& config) {
    const double dx = x - config.center_x;
    const double dy = y - config.center_y;
    return ((dx * dx) + (dy * dy) <= (config.circle_radius * config.circle_radius)) ? 1 : 0;
}

std::vector<Example> generate_balanced_dataset(int sample_count, const Config& config, int seed) {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<double> dist(config.coordinate_min, config.coordinate_max);

    const int positives_needed = sample_count / 2;
    const int negatives_needed = sample_count - positives_needed;
    int positives = 0;
    int negatives = 0;
    std::vector<Example> examples;
    examples.reserve(sample_count);

    while (positives < positives_needed || negatives < negatives_needed) {
        const double x = dist(rng);
        const double y = dist(rng);
        const int label = classify_point(x, y, config);
        if (label == 1 && positives < positives_needed) {
            examples.push_back({{x, y}, label});
            ++positives;
        } else if (label == 0 && negatives < negatives_needed) {
            examples.push_back({{x, y}, label});
            ++negatives;
        }
    }

    std::shuffle(examples.begin(), examples.end(), rng);
    return examples;
}

ForwardPass forward_pass(const NeuralNetwork& model, const std::vector<double>& features) {
    ForwardPass pass;
    pass.hidden_z.assign(model.hidden_size, 0.0);
    pass.hidden_a.assign(model.hidden_size, 0.0);

    for (int i = 0; i < model.hidden_size; ++i) {
        double z = model.b1[i];
        for (int j = 0; j < model.input_size; ++j) {
            z += model.w1[i][j] * features[j];
        }
        pass.hidden_z[i] = z;
        pass.hidden_a[i] = activation_forward(z, model.hidden_activation);
    }

    pass.output_z = model.b2;
    for (int i = 0; i < model.hidden_size; ++i) {
        pass.output_z += model.w2[i] * pass.hidden_a[i];
    }
    pass.output_y = sigmoid(pass.output_z);
    return pass;
}

std::pair<double, double> evaluate_model(const NeuralNetwork& model, const std::vector<Example>& examples, const Config& config) {
    double total_loss = 0.0;
    int correct = 0;
    for (const Example& example : examples) {
        const ForwardPass pass = forward_pass(model, example.features);
        total_loss += data_loss(pass.output_y, example.label, config.loss_function);
        const int prediction = pass.output_y >= 0.5 ? 1 : 0;
        if (prediction == example.label) {
            ++correct;
        }
    }
    return {total_loss / examples.size(), static_cast<double>(correct) / examples.size()};
}

void print_vector(const std::vector<double>& values, int precision = 4) {
    std::cout << "[";
    for (std::size_t index = 0; index < values.size(); ++index) {
        if (index > 0) {
            std::cout << ", ";
        }
        std::cout << std::fixed << std::setprecision(precision) << values[index];
    }
    std::cout << "]";
}

void train_model(NeuralNetwork& model, const std::vector<Example>& train_data, const std::vector<Example>& test_data, const Config& config) {
    for (int epoch = 1; epoch <= config.epochs; ++epoch) {
        for (const Example& example : train_data) {
            const ForwardPass pass = forward_pass(model, example.features);
            const double delta_out = output_delta(pass.output_y, example.label, config.loss_function);

            std::vector<double> hidden_delta(model.hidden_size, 0.0);
            for (int i = 0; i < model.hidden_size; ++i) {
                hidden_delta[i] = activation_derivative(pass.hidden_z[i], pass.hidden_a[i], model.hidden_activation) * model.w2[i] * delta_out;
            }

            for (int i = 0; i < model.hidden_size; ++i) {
                model.w2[i] -= config.learning_rate * delta_out * pass.hidden_a[i];
            }
            model.b2 -= config.learning_rate * delta_out;

            for (int i = 0; i < model.hidden_size; ++i) {
                for (int j = 0; j < model.input_size; ++j) {
                    model.w1[i][j] -= config.learning_rate * hidden_delta[i] * example.features[j];
                }
                model.b1[i] -= config.learning_rate * hidden_delta[i];
            }
        }

        if (epoch == 1 || epoch == config.epochs || epoch % 20 == 0) {
            const auto train_metrics = evaluate_model(model, train_data, config);
            const auto test_metrics = evaluate_model(model, test_data, config);
            std::cout << "Epoch " << std::setw(3) << epoch << "/" << config.epochs
                      << " | train loss " << std::fixed << std::setprecision(4) << train_metrics.first
                      << " | train acc " << std::setprecision(2) << (train_metrics.second * 100.0) << "%"
                      << " | test acc " << (test_metrics.second * 100.0) << "%\n";
        }
    }
}

int main() {
    Config config;
    auto train_data = generate_balanced_dataset(config.train_samples, config, config.seed);
    auto test_data = generate_balanced_dataset(config.test_samples, config, config.seed + 1);
    NeuralNetwork model = init_model(config);

    train_model(model, train_data, test_data, config);

    const auto train_metrics = evaluate_model(model, train_data, config);
    const auto test_metrics = evaluate_model(model, test_data, config);
    const ForwardPass sample = forward_pass(model, test_data.front().features);

    std::cout << "\nFinal train loss: " << std::fixed << std::setprecision(4) << train_metrics.first << "\n";
    std::cout << "Final train accuracy: " << std::setprecision(2) << (train_metrics.second * 100.0) << "%\n";
    std::cout << "Final test loss: " << std::setprecision(4) << test_metrics.first << "\n";
    std::cout << "Final test accuracy: " << std::setprecision(2) << (test_metrics.second * 100.0) << "%\n";
    std::cout << "Hidden activation: " << config.hidden_activation << "\n";
    std::cout << "Loss function: " << config.loss_function << "\n";
    std::cout << "Optimization: gradient_descent\n";
    std::cout << "Sample forward pass:\n";
    std::cout << "  input = [" << test_data.front().features[0] << ", " << test_data.front().features[1] << "]\n";
    std::cout << "  hidden_z = ";
    print_vector(sample.hidden_z);
    std::cout << "\n";
    std::cout << "  hidden_a = ";
    print_vector(sample.hidden_a);
    std::cout << "\n";
    std::cout << "  output_z = " << std::setprecision(4) << sample.output_z << "\n";
    std::cout << "  y = " << std::setprecision(4) << sample.output_y << "\n";
    std::cout << "  label = " << test_data.front().label << "\n";
    return 0;
}
