%% optimize_FLF_parameters.m
% Find optimal alpha values and polynomial order for maximum accuracy

clear; clc; close all;

%% Set paths
gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

%% Define search space
n_values = [6, 8, 10, 12];  % Polynomial orders to test
alpha_ranges = {
    [0.5, 1.0, 1.5],           % Original
    [0.3, 0.7, 1.0, 1.3],      % Finer spacing
    [0.5, 1.0, 2.0],           % Wider range
    [0.2, 0.5, 1.0, 1.5, 2.0], % More scales
    [0.8, 1.0, 1.2],           % Around standard
    [0.4, 0.8, 1.2, 1.6]       % Different spacing
};

image_size = [64, 64];
train_ratio = 0.8;

%% Storage for results
results = [];
result_idx = 1;

%% Grid search
fprintf('OPTIMIZING FRACTIONAL LEGENDRE PARAMETERS\n');
fprintf('==========================================\n\n');

total_combinations = length(n_values) * length(alpha_ranges);
current_combination = 0;

best_accuracy = 0;
best_n = 0;
best_alphas = [];

for n = n_values
    for alpha_idx = 1:length(alpha_ranges)
        current_combination = current_combination + 1;
        alpha_scales = alpha_ranges{alpha_idx};
        
        fprintf('[%d/%d] Testing n=%d, alphas=[%s]...\n', ...
            current_combination, total_combinations, n, num2str(alpha_scales));
        
        try
            % Extract features
            tic;
            [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, ...
                n, alpha_scales, image_size);
            feature_time = toc;
            
            % Split data
            rng(42);  % Fixed seed
            [X_train, X_test, y_train, y_test, ~, ~] = split_data_fixed(features, labels, train_ratio);
            
            % Quick SVM training (no optimization for speed)
            tic;
            svm_model = fitcsvm(X_train, y_train, ...
                'KernelFunction', 'rbf', ...
                'Standardize', true, ...
                'BoxConstraint', 10, ...
                'KernelScale', 'auto');
            train_time = toc;
            
            % Evaluate
            [accuracy, precision, recall, f1, ~] = evaluate_classifier(svm_model, X_test, y_test);
            
            % Store results
            results(result_idx).n = n;
            results(result_idx).alphas = alpha_scales;
            results(result_idx).num_features = size(features, 2);
            results(result_idx).accuracy = accuracy;
            results(result_idx).precision = precision;
            results(result_idx).recall = recall;
            results(result_idx).f1 = f1;
            results(result_idx).feature_time = feature_time;
            results(result_idx).train_time = train_time;
            result_idx = result_idx + 1;
            
            fprintf('  Accuracy: %.4f, Features: %d, Time: %.1fs\n', ...
                accuracy, size(features, 2), feature_time + train_time);
            
            % Update best
            if accuracy > best_accuracy
                best_accuracy = accuracy;
                best_n = n;
                best_alphas = alpha_scales;
            end
            
        catch ME
            fprintf('  Error: %s\n', ME.message);
        end
    end
end

%% Display results table
fprintf('\n\n=== OPTIMIZATION RESULTS ===\n');
fprintf('%-4s | %-30s | %-8s | %-8s | %-8s\n', 'n', 'Alpha Values', 'Features', 'Accuracy', 'F1-Score');
fprintf('%s\n', repmat('-', 70, 1));

[~, sorted_idx] = sort([results.accuracy], 'descend');
for i = 1:min(10, length(results))  % Show top 10
    r = results(sorted_idx(i));
    fprintf('%-4d | %-30s | %-8d | %-8.4f | %-8.4f\n', ...
        r.n, sprintf('[%s]', num2str(r.alphas)), r.num_features, r.accuracy, r.f1);
end

%% Best configuration
fprintf('\n*** OPTIMAL CONFIGURATION ***\n');
fprintf('Polynomial order (n): %d\n', best_n);
fprintf('Alpha values: [%s]\n', num2str(best_alphas));
fprintf('Accuracy: %.4f (%.2f%%)\n', best_accuracy, best_accuracy*100);

%% Fine-tune around best
fprintf('\n\nFINE-TUNING AROUND BEST CONFIGURATION...\n');

% Generate alpha variations around best
if length(best_alphas) > 1
    alpha_variations = [];
    for i = 1:5
        % Small perturbations
        perturbation = (rand(size(best_alphas)) - 0.5) * 0.2;
        new_alphas = best_alphas + perturbation;
        new_alphas = max(0.1, new_alphas);  % Keep positive
        alpha_variations{i} = new_alphas;
    end
    
    % Test variations
    for i = 1:length(alpha_variations)
        alpha_scales = alpha_variations{i};
        fprintf('Testing alphas=[%s]...', num2str(alpha_scales, '%.2f '));
        
        [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, ...
            best_n, alpha_scales, image_size);
        
        rng(42);
        [X_train, X_test, y_train, y_test, ~, ~] = split_data_fixed(features, labels, train_ratio);
        
        % Train with optimization
        svm_model = train_svm_optimized_FLF(X_train, y_train);
        [accuracy, ~, ~, ~, ~] = evaluate_classifier(svm_model, X_test, y_test);
        
        fprintf(' Accuracy: %.4f\n', accuracy);
        
        if accuracy > best_accuracy
            best_accuracy = accuracy;
            best_alphas = alpha_scales;
            fprintf('  NEW BEST!\n');
        end
    end
end

%% Save optimal configuration
fprintf('\n\n=== FINAL OPTIMAL CONFIGURATION ===\n');
fprintf('Polynomial order (n): %d\n', best_n);
fprintf('Alpha values: [%s]\n', num2str(best_alphas, '%.3f '));
fprintf('Best Accuracy: %.4f (%.2f%%)\n', best_accuracy, best_accuracy*100);

save('optimal_FLF_parameters.mat', 'best_n', 'best_alphas', 'best_accuracy', 'results');
fprintf('\nOptimal parameters saved to optimal_FLF_parameters.mat\n');

%% Visualization
figure('Position', [100, 100, 1200, 400]);

% Plot 1: Accuracy vs n for different alpha configs
subplot(1,3,1);
n_unique = unique([results.n]);
for n = n_unique
    n_results = results([results.n] == n);
    accuracies = [n_results.accuracy];
    plot(1:length(accuracies), accuracies, 'o-', 'LineWidth', 2, 'DisplayName', sprintf('n=%d', n));
    hold on;
end
xlabel('Configuration Index');
ylabel('Accuracy');
title('Accuracy by Polynomial Order');
legend('Location', 'best');
grid on;

% Plot 2: Feature count vs accuracy
subplot(1,3,2);
scatter([results.num_features], [results.accuracy]*100, 50, 'filled');
xlabel('Number of Features');
ylabel('Accuracy (%)');
title('Accuracy vs Feature Count');
grid on;

% Plot 3: Best alpha analysis
subplot(1,3,3);
if length(best_alphas) <= 5
    bar(best_alphas);
    xlabel('Alpha Index');
    ylabel('Alpha Value');
    title('Optimal Alpha Values');
else
    plot(best_alphas, 'o-', 'LineWidth', 2);
    xlabel('Alpha Index');
    ylabel('Alpha Value');
    title('Optimal Alpha Values');
end
grid on;

sgtitle('Fractional Legendre Parameter Optimization');

%% Helper functions (copy from main code)
function svm_model = train_svm_optimized_FLF(X_train, y_train)
    rng(42);
    box_values = [1, 10, 100];
    kernel_scales = {'auto'};
    
    best_acc = 0;
    best_box = 10;
    
    for box = box_values
        temp_model = fitcsvm(X_train, y_train, ...
            'KernelFunction', 'rbf', ...
            'Standardize', true, ...
            'BoxConstraint', box, ...
            'KernelScale', 'auto', ...
            'KFold', 3);
        
        cv_acc = 1 - kfoldLoss(temp_model);
        if cv_acc > best_acc
            best_acc = cv_acc;
            best_box = box;
        end
    end
    
    svm_model = fitcsvm(X_train, y_train, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'BoxConstraint', best_box, ...
        'KernelScale', 'auto');
end

% Include other helper functions from main_gun_detection_FLF.m...