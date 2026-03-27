function own_ai_model_demo()
config.seed = 7;
config.train_samples = 900;
config.test_samples = 300;
config.coordinate_min = -1.0;
config.coordinate_max = 1.0;
config.circle_radius = 0.78;
config.center_x = 0.2;
config.center_y = -0.1;
config.hidden_size = 10;
config.epochs = 180;
config.learning_rate = 0.08;
config.hidden_activation = "relu";
config.loss_function = "binary_cross_entropy";

rng(config.seed);
[train_x, train_y] = generate_balanced_dataset(config.train_samples, config, config.seed);
[test_x, test_y] = generate_balanced_dataset(config.test_samples, config, config.seed + 1);
model = init_model(config);

for epoch = 1:config.epochs
    for i = 1:size(train_x, 1)
        features = train_x(i, :)';
        label = train_y(i);

        [hidden_z, hidden_a, ~, prob] = forward_pass(model, features);
        delta_out = output_delta(prob, label, config.loss_function);

        grad_w2 = delta_out .* hidden_a;
        grad_b2 = delta_out;

        hidden_delta = zeros(config.hidden_size, 1);
        for h = 1:config.hidden_size
            hidden_delta(h) = hidden_activation_derivative(hidden_z(h), hidden_a(h), config.hidden_activation) * model.w2(h) * delta_out;
        end

        model.w2 = model.w2 - config.learning_rate .* grad_w2;
        model.b2 = model.b2 - config.learning_rate .* grad_b2;

        for h = 1:config.hidden_size
            model.w1(h, :) = model.w1(h, :) - config.learning_rate .* hidden_delta(h) .* train_x(i, :);
            model.b1(h) = model.b1(h) - config.learning_rate .* hidden_delta(h);
        end
    end

    if epoch == 1 || epoch == config.epochs || mod(epoch, 20) == 0
        [train_loss, train_acc] = evaluate_model(model, train_x, train_y, config);
        [~, test_acc] = evaluate_model(model, test_x, test_y, config);
        fprintf('Epoch %3d/%d | train loss %.4f | train acc %.2f%% | test acc %.2f%%\n', ...
            epoch, config.epochs, train_loss, train_acc * 100.0, test_acc * 100.0);
    end
end

[train_loss, train_acc] = evaluate_model(model, train_x, train_y, config);
[test_loss, test_acc] = evaluate_model(model, test_x, test_y, config);
[hidden_z, hidden_a, output_z, output_y] = forward_pass(model, test_x(1, :)');

fprintf('\nFinal train loss: %.4f\n', train_loss);
fprintf('Final train accuracy: %.2f%%\n', train_acc * 100.0);
fprintf('Final test loss: %.4f\n', test_loss);
fprintf('Final test accuracy: %.2f%%\n', test_acc * 100.0);
fprintf('Hidden activation: %s\n', config.hidden_activation);
fprintf('Loss function: %s\n', config.loss_function);
fprintf('Optimization: gradient_descent\n');
disp('Sample forward pass:');
disp(['  input = ', mat2str(test_x(1, :), 4)]);
disp(['  hidden_z = ', mat2str(hidden_z', 4)]);
disp(['  hidden_a = ', mat2str(hidden_a', 4)]);
disp(['  output_z = ', num2str(output_z, 4)]);
disp(['  y = ', num2str(output_y, 4)]);
disp(['  label = ', num2str(test_y(1))]);
end

function model = init_model(config)
first_scale = 1.0 / sqrt(2.0);
second_scale = 1.0 / sqrt(config.hidden_size);
model.w1 = (rand(config.hidden_size, 2) * 2 * first_scale) - first_scale;
model.b1 = zeros(config.hidden_size, 1);
model.w2 = (rand(config.hidden_size, 1) * 2 * second_scale) - second_scale;
model.b2 = 0.0;
end

function [features, labels] = generate_balanced_dataset(sample_count, config, seed)
rng(seed);
features = zeros(sample_count, 2);
labels = zeros(sample_count, 1);
positives_needed = floor(sample_count / 2);
negatives_needed = sample_count - positives_needed;
positives = 0;
negatives = 0;
index = 1;

while positives < positives_needed || negatives < negatives_needed
    x = rand() * (config.coordinate_max - config.coordinate_min) + config.coordinate_min;
    y = rand() * (config.coordinate_max - config.coordinate_min) + config.coordinate_min;
    label = classify_point(x, y, config);
    if label == 1 && positives < positives_needed
        features(index, :) = [x, y];
        labels(index) = label;
        positives = positives + 1;
        index = index + 1;
    elseif label == 0 && negatives < negatives_needed
        features(index, :) = [x, y];
        labels(index) = label;
        negatives = negatives + 1;
        index = index + 1;
    end
end
end

function label = classify_point(x, y, config)
dx = x - config.center_x;
dy = y - config.center_y;
label = ((dx * dx) + (dy * dy) <= config.circle_radius ^ 2);
end

function [hidden_z, hidden_a, output_z, output_y] = forward_pass(model, features)
hidden_z = model.w1 * features + model.b1;
hidden_a = zeros(size(hidden_z));
for i = 1:length(hidden_z)
    hidden_a(i) = activation_forward(hidden_z(i), model.hidden_activation);
end
output_z = model.w2' * hidden_a + model.b2;
output_y = sigmoid(output_z);
end

function value = activation_forward(x, activation_name)
if activation_name == "relu"
    value = max(0.0, x);
elseif activation_name == "sigmoid"
    value = sigmoid(x);
else
    error('Unsupported activation.');
end
end

function value = hidden_activation_derivative(z, a, activation_name)
if activation_name == "relu"
    value = double(z > 0.0);
elseif activation_name == "sigmoid"
    value = a * (1.0 - a);
else
    error('Unsupported activation.');
end
end

function value = sigmoid(x)
bounded = min(max(x, -60.0), 60.0);
value = 1.0 / (1.0 + exp(-bounded));
end

function value = data_loss(prob, label, loss_function)
safe_prob = min(max(prob, 1e-7), 1.0 - 1e-7);
if loss_function == "binary_cross_entropy"
    value = -((label * log(safe_prob)) + ((1 - label) * log(1.0 - safe_prob)));
elseif loss_function == "mean_squared_error"
    value = (prob - label) ^ 2;
else
    error('Unsupported loss function.');
end
end

function value = output_delta(prob, label, loss_function)
if loss_function == "binary_cross_entropy"
    value = prob - label;
elseif loss_function == "mean_squared_error"
    value = 2.0 * (prob - label) * prob * (1.0 - prob);
else
    error('Unsupported loss function.');
end
end

function [loss, accuracy] = evaluate_model(model, features, labels, config)
correct = 0;
total_loss = 0.0;
for i = 1:size(features, 1)
    [~, ~, ~, prob] = forward_pass(model, features(i, :)');
    total_loss = total_loss + data_loss(prob, labels(i), config.loss_function);
    prediction = double(prob >= 0.5);
    if prediction == labels(i)
        correct = correct + 1;
    end
end
loss = total_loss / size(features, 1);
accuracy = correct / size(features, 1);
end

own_ai_model_demo();
