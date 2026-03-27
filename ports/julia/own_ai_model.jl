using Random
using Printf

struct Example
    features::Vector{Float64}
    label::Int
end

Base.@kwdef struct Config
    seed::Int = 7
    train_samples::Int = 900
    test_samples::Int = 300
    coordinate_min::Float64 = -1.0
    coordinate_max::Float64 = 1.0
    circle_radius::Float64 = 0.78
    center_x::Float64 = 0.2
    center_y::Float64 = -0.1
    hidden_size::Int = 10
    epochs::Int = 180
    learning_rate::Float64 = 0.08
    hidden_activation::String = "relu"
    loss_function::String = "binary_cross_entropy"
end

mutable struct NeuralNetwork
    input_size::Int
    hidden_size::Int
    hidden_activation::String
    w1::Matrix{Float64}
    b1::Vector{Float64}
    w2::Vector{Float64}
    b2::Float64
end

relu(x) = max(0.0, x)

function sigmoid(x::Float64)
    bounded = clamp(x, -60.0, 60.0)
    return 1.0 / (1.0 + exp(-bounded))
end

function activation_forward(x::Float64, name::String)
    if name == "relu"
        return relu(x)
    elseif name == "sigmoid"
        return sigmoid(x)
    else
        error("Unsupported activation: $name")
    end
end

function activation_derivative(z::Float64, a::Float64, name::String)
    if name == "relu"
        return z > 0.0 ? 1.0 : 0.0
    elseif name == "sigmoid"
        return a * (1.0 - a)
    else
        error("Unsupported activation: $name")
    end
end

function init_network(config::Config)
    rng = MersenneTwister(config.seed)
    first_scale = 1.0 / sqrt(2.0)
    second_scale = 1.0 / sqrt(config.hidden_size)
    w1 = rand(rng, config.hidden_size, 2) .* (2 * first_scale) .- first_scale
    b1 = zeros(config.hidden_size)
    w2 = rand(rng, config.hidden_size) .* (2 * second_scale) .- second_scale
    b2 = 0.0
    return NeuralNetwork(2, config.hidden_size, config.hidden_activation, w1, b1, w2, b2)
end

function classify_point(x::Float64, y::Float64, config::Config)
    dx = x - config.center_x
    dy = y - config.center_y
    return ((dx * dx) + (dy * dy) <= config.circle_radius ^ 2) ? 1 : 0
end

function generate_balanced_dataset(sample_count::Int, config::Config, seed::Int)
    rng = MersenneTwister(seed)
    positives_needed = div(sample_count, 2)
    negatives_needed = sample_count - positives_needed
    positives = 0
    negatives = 0
    examples = Example[]

    while positives < positives_needed || negatives < negatives_needed
        x = rand(rng) * (config.coordinate_max - config.coordinate_min) + config.coordinate_min
        y = rand(rng) * (config.coordinate_max - config.coordinate_min) + config.coordinate_min
        label = classify_point(x, y, config)
        if label == 1 && positives < positives_needed
            push!(examples, Example([x, y], label))
            positives += 1
        elseif label == 0 && negatives < negatives_needed
            push!(examples, Example([x, y], label))
            negatives += 1
        end
    end

    shuffle!(rng, examples)
    return examples
end

function forward_pass(model::NeuralNetwork, features::Vector{Float64})
    hidden_z = zeros(model.hidden_size)
    hidden_a = zeros(model.hidden_size)
    for i in 1:model.hidden_size
        z = model.b1[i]
        for j in 1:model.input_size
            z += model.w1[i, j] * features[j]
        end
        hidden_z[i] = z
        hidden_a[i] = activation_forward(z, model.hidden_activation)
    end

    output_z = model.b2
    for i in 1:model.hidden_size
        output_z += model.w2[i] * hidden_a[i]
    end
    output_y = sigmoid(output_z)
    return hidden_z, hidden_a, output_z, output_y
end

function data_loss(prob::Float64, label::Int, loss_name::String)
    safe_prob = clamp(prob, 1e-7, 1.0 - 1e-7)
    if loss_name == "binary_cross_entropy"
        return -((label * log(safe_prob)) + ((1 - label) * log(1.0 - safe_prob)))
    elseif loss_name == "mean_squared_error"
        return (prob - label) ^ 2
    else
        error("Unsupported loss function: $loss_name")
    end
end

function output_delta(prob::Float64, label::Int, loss_name::String)
    if loss_name == "binary_cross_entropy"
        return prob - label
    elseif loss_name == "mean_squared_error"
        return 2.0 * (prob - label) * prob * (1.0 - prob)
    else
        error("Unsupported loss function: $loss_name")
    end
end

function train!(model::NeuralNetwork, train_data::Vector{Example}, test_data::Vector{Example}, config::Config)
    for epoch in 1:config.epochs
        for example in train_data
            hidden_z, hidden_a, _, prob = forward_pass(model, example.features)
            delta_out = output_delta(prob, example.label, config.loss_function)

            grad_w2 = delta_out .* hidden_a
            grad_b2 = delta_out
            hidden_delta = zeros(model.hidden_size)
            for i in 1:model.hidden_size
                hidden_delta[i] = activation_derivative(
                    hidden_z[i],
                    hidden_a[i],
                    model.hidden_activation,
                ) * model.w2[i] * delta_out
            end

            for i in 1:model.hidden_size
                model.w2[i] -= config.learning_rate * grad_w2[i]
            end
            model.b2 -= config.learning_rate * grad_b2

            for i in 1:model.hidden_size
                for j in 1:model.input_size
                    model.w1[i, j] -= config.learning_rate * hidden_delta[i] * example.features[j]
                end
                model.b1[i] -= config.learning_rate * hidden_delta[i]
            end
        end

        if epoch == 1 || epoch == config.epochs || epoch % 20 == 0
            train_loss, train_acc = evaluate(model, train_data, config)
            _, test_acc = evaluate(model, test_data, config)
            @printf(
                "Epoch %3d/%d | train loss %.4f | train acc %.2f%% | test acc %.2f%%\n",
                epoch,
                config.epochs,
                train_loss,
                train_acc * 100.0,
                test_acc * 100.0,
            )
        end
    end
end

function evaluate(model::NeuralNetwork, examples::Vector{Example}, config::Config)
    total_loss = 0.0
    correct = 0
    for example in examples
        _, _, _, prob = forward_pass(model, example.features)
        total_loss += data_loss(prob, example.label, config.loss_function)
        predicted = prob >= 0.5 ? 1 : 0
        correct += predicted == example.label ? 1 : 0
    end
    return total_loss / length(examples), correct / length(examples)
end

function main()
    config = Config()
    train_data = generate_balanced_dataset(config.train_samples, config, config.seed)
    test_data = generate_balanced_dataset(config.test_samples, config, config.seed + 1)
    model = init_network(config)

    train!(model, train_data, test_data, config)

    train_loss, train_acc = evaluate(model, train_data, config)
    test_loss, test_acc = evaluate(model, test_data, config)
    sample = test_data[1]
    hidden_z, hidden_a, output_z, output_y = forward_pass(model, sample.features)

    println()
    @printf("Final train loss: %.4f\n", train_loss)
    @printf("Final train accuracy: %.2f%%\n", train_acc * 100.0)
    @printf("Final test loss: %.4f\n", test_loss)
    @printf("Final test accuracy: %.2f%%\n", test_acc * 100.0)
    println("Hidden activation: $(config.hidden_activation)")
    println("Loss function: $(config.loss_function)")
    println("Optimization: gradient_descent")
    println("Sample forward pass:")
    println("  input = $(sample.features)")
    println("  hidden_z = $(round.(hidden_z, digits=4))")
    println("  hidden_a = $(round.(hidden_a, digits=4))")
    println("  output_z = $(round(output_z, digits=4))")
    println("  y = $(round(output_y, digits=4))")
    println("  label = $(sample.label)")
end

main()
