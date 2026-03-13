%% svm_FLF_seed_analysis.m
% Test Fractional Legendre Features SVM with different random seeds (20 to 60)
% Analyze stability and performance consistency of FLF approach

clear; clc; 

%% Parameters
n = 10;  % Number of Legendre polynomials
alpha_scales = [0.5, 1.0, 1.5];  % Multi-scale fractional orders
image_size = [64, 64];  % Standard image size
train_ratio = 0.8;  % 80% for training, 20% for testing

%% Seed range
seed_start = 20;
seed_end = 60;
num_seeds = seed_end - seed_start + 1;

%% Set data paths
gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

fprintf('FRACTIONAL LEGENDRE SVM - Stability Analysis (Seeds %d to %d)\n', seed_start, seed_end);
fprintf('==========================================================\n');
fprintf('Alpha scales: [%s]\n', num2str(alpha_scales));
fprintf('Total features per image: %d\n\n', n*n*length(alpha_scales));

%% Load and process images with FLF features
fprintf('Loading images and extracting FLF features...\n');
tic;
[features_flf, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);
loading_time = toc;

fprintf('Total samples: %d\n', length(labels));
fprintf('Gun samples: %d\n', sum(labels == 1));
fprintf('No-gun samples: %d\n', sum(labels == 0));
fprintf('Feature extraction time: %.2f seconds\n', loading_time);
fprintf('Feature vector size: %d\n\n', size(features_flf, 2));

%% Initialize results storage
results_flf = zeros(num_seeds, 5);  % seed, accuracy, precision, recall, f1
training_times = zeros(num_seeds, 1);

%% Test each seed
fprintf('Testing FLF method with different seeds...\n');
fprintf('Seed | Accuracy | Precision | Recall  | F1-Score | Train Time (s)\n');
fprintf('-----|----------|-----------|---------|----------|---------------\n');

for i = 1:num_seeds
    seed = seed_start + i - 1;
    
    % Set random seed
    rng(seed);
    
    % Split data with current seed
    n_samples = size(features_flf, 1);
    idx = randperm(n_samples);
    n_train = round(n_samples * train_ratio);
    train_idx = idx(1:n_train);
    test_idx = idx(n_train+1:end);
    
    % Prepare training and testing sets
    X_train = features_flf(train_idx, :);
    X_test = features_flf(test_idx, :);
    y_train = labels(train_idx);
    y_test = labels(test_idx);
    
    % Train FLF SVM
    tic;
    svm_flf = fitcsvm(X_train, y_train, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'BoxConstraint', 10, ...
        'KernelScale', 'auto');
    training_times(i) = toc;
    
    % Evaluate FLF
    y_pred = predict(svm_flf, X_test);
    [acc, prec, rec, f1, ~] = calculate_metrics(y_test, y_pred);
    results_flf(i, :) = [seed, acc, prec, rec, f1];
    
    % Display results
    fprintf(' %3d |  %.4f  |   %.4f   |  %.4f  |  %.4f   |     %.3f\n', ...
        seed, acc, prec, rec, f1, training_times(i));
end

%% Calculate statistics
fprintf('\n==========================================================\n');
fprintf('STATISTICAL SUMMARY - FRACTIONAL LEGENDRE FEATURES\n');
fprintf('==========================================================\n\n');

% Calculate mean and std for each metric
metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
for i = 1:4
    metric_values = results_flf(:, i+1);
    
    mean_val = mean(metric_values);
    std_val = std(metric_values);
    min_val = min(metric_values);
    max_val = max(metric_values);
    
    fprintf('%s:\n', metrics{i});
    fprintf('  Mean:       %.4f ± %.4f\n', mean_val, std_val);
    fprintf('  Range:      %.4f to %.4f\n', min_val, max_val);
    fprintf('  Stability:  %.2f%% coefficient of variation\n\n', (std_val/mean_val)*100);
end

%% Find best and worst performing seeds
[best_f1, best_idx] = max(results_flf(:, 5));
best_seed = results_flf(best_idx, 1);
best_results = results_flf(best_idx, :);

[worst_f1, worst_idx] = min(results_flf(:, 5));
worst_seed = results_flf(worst_idx, 1);
worst_results = results_flf(worst_idx, :);

fprintf('==========================================================\n');
fprintf('SEED PERFORMANCE ANALYSIS\n');
fprintf('==========================================================\n');
fprintf('Best Performing Seed: %d\n', best_seed);
fprintf('  Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f\n\n', ...
    best_results(2), best_results(3), best_results(4), best_results(5));

fprintf('Worst Performing Seed: %d\n', worst_seed);
fprintf('  Accuracy: %.4f, Precision: %.4f, Recall: %.4f, F1: %.4f\n\n', ...
    worst_results(2), worst_results(3), worst_results(4), worst_results(5));

fprintf('Performance Range:\n');
fprintf('  F1-Score: %.4f (%.2f%% variation)\n', ...
    max(results_flf(:,5)) - min(results_flf(:,5)), ...
    ((max(results_flf(:,5)) - min(results_flf(:,5))) / mean(results_flf(:,5))) * 100);

%% Training time analysis
fprintf('\nTraining Time Analysis:\n');
fprintf('  Average: %.3f ± %.3f seconds\n', mean(training_times), std(training_times));
fprintf('  Range: %.3f to %.3f seconds\n', min(training_times), max(training_times));

%% Visualize results
figure;

% Plot 1: Performance metrics across seeds
subplot(2,3,1);
plot(results_flf(:,1), results_flf(:,2), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(results_flf(:,1), results_flf(:,5), 'r-s', 'LineWidth', 2, 'MarkerSize', 6);
xlabel('Random Seed');
ylabel('Score');
title('Accuracy and F1-Score Across Seeds');
legend('Accuracy', 'F1-Score', 'Location', 'best');
grid on;
ylim([0.85, 1.0]);

% Plot 2: All metrics comparison
subplot(2,3,2);
plot(results_flf(:,1), results_flf(:,2:5), 'LineWidth', 1.5);
xlabel('Random Seed');
ylabel('Score');
title('All Performance Metrics');
legend(metrics, 'Location', 'best');
grid on;
ylim([0.85, 1.0]);

% Plot 3: Performance distribution
subplot(2,3,3);
boxplot(results_flf(:,2:5), 'Labels', metrics);
ylabel('Score');
title('Performance Distribution');
grid on;

% Plot 4: Training time across seeds
subplot(2,3,4);
bar(results_flf(:,1), training_times, 'FaceColor', [0.3 0.7 0.9]);
xlabel('Random Seed');
ylabel('Training Time (seconds)');
title('Training Time Across Seeds');
grid on;

% Plot 5: Stability analysis (coefficient of variation)
subplot(2,3,5);
cv_values = zeros(4,1);
for i = 1:4
    cv_values(i) = (std(results_flf(:,i+1)) / mean(results_flf(:,i+1))) * 100;
end
bar(cv_values, 'FaceColor', [0.8 0.4 0.2]);
set(gca, 'XTickLabel', metrics);
ylabel('Coefficient of Variation (%)');
title('Stability Analysis (Lower is Better)');
grid on;

% Plot 6: Individual scale contribution analysis (using best seed)
subplot(2,3,6);
rng(best_seed);
n_samples = size(features_flf, 1);
idx = randperm(n_samples);
n_train = round(n_samples * train_ratio);
train_idx = idx(1:n_train);
test_idx = idx(n_train+1:end);

scale_accs = zeros(length(alpha_scales), 1);
for a = 1:length(alpha_scales)
    % Extract single-scale features
    [features_single, ~] = extract_features_single_scale(gun_folder, no_gun_folder, n, alpha_scales(a), image_size);
    X_train_s = features_single(train_idx, :);
    X_test_s = features_single(test_idx, :);
    y_train = labels(train_idx);
    y_test = labels(test_idx);
    
    svm_s = fitcsvm(X_train_s, y_train, 'KernelFunction', 'rbf', 'Standardize', true);
    y_pred_s = predict(svm_s, X_test_s);
    scale_accs(a) = sum(y_pred_s == y_test) / length(y_test);
end

bar(alpha_scales, scale_accs, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Alpha (α)');
ylabel('Accuracy');
title(sprintf('Individual Scale Performance (Seed %d)', best_seed));
ylim([0.8, 1.0]);
grid on;

sgtitle('Fractional Legendre Features: Stability Analysis Across Seeds');

%% Save results
save('flf_stability_results.mat', 'results_flf', 'training_times', 'alpha_scales', 'best_seed', 'metrics');
fprintf('\nResults saved to flf_stability_results.mat\n');

%% Create detailed report
report_file = 'flf_stability_report.txt';
fid = fopen(report_file, 'w');

fprintf(fid, 'Fractional Legendre Features Stability Analysis Report\n');
fprintf(fid, '=====================================================\n\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Seeds tested: %d to %d (%d total)\n', seed_start, seed_end, num_seeds);
fprintf(fid, 'Alpha scales: [%s]\n', num2str(alpha_scales));
fprintf(fid, 'Feature vector size: %d\n\n', size(features_flf, 2));

fprintf(fid, 'OVERALL PERFORMANCE:\n');
fprintf(fid, 'Accuracy:    %.4f ± %.4f (%.2f%% to %.2f%%)\n', ...
    mean(results_flf(:,2)), std(results_flf(:,2)), ...
    min(results_flf(:,2))*100, max(results_flf(:,2))*100);
fprintf(fid, 'Precision:   %.4f ± %.4f\n', mean(results_flf(:,3)), std(results_flf(:,3)));
fprintf(fid, 'Recall:      %.4f ± %.4f\n', mean(results_flf(:,4)), std(results_flf(:,4)));
fprintf(fid, 'F1-Score:    %.4f ± %.4f\n\n', mean(results_flf(:,5)), std(results_flf(:,5)));

fprintf(fid, 'STABILITY METRICS:\n');
fprintf(fid, 'Accuracy CV:     %.2f%%\n', (std(results_flf(:,2))/mean(results_flf(:,2)))*100);
fprintf(fid, 'F1-Score CV:     %.2f%%\n', (std(results_flf(:,5))/mean(results_flf(:,5)))*100);
fprintf(fid, 'Training Time:   %.3f ± %.3f seconds\n\n', mean(training_times), std(training_times));

fprintf(fid, 'RECOMMENDATIONS:\n');
fprintf(fid, 'Best seed for reproduction: %d\n', best_seed);
fprintf(fid, 'Expected performance: %.2f%% ± %.2f%% accuracy\n', ...
    mean(results_flf(:,2))*100, std(results_flf(:,2))*100);
fprintf(fid, 'Alpha configuration: [%s] provides good multi-scale representation\n', num2str(alpha_scales));

fclose(fid);
fprintf('Detailed report saved to %s\n', report_file);

%% Final summary
fprintf('\n==========================================================\n');
fprintf('FINAL SUMMARY - FRACTIONAL LEGENDRE FEATURES\n');
fprintf('==========================================================\n');
fprintf('Performance Consistency:\n');
fprintf('  Average Accuracy: %.2f%% ± %.2f%%\n', mean(results_flf(:,2))*100, std(results_flf(:,2))*100);
fprintf('  Average F1-Score: %.4f ± %.4f\n', mean(results_flf(:,5)), std(results_flf(:,5)));
fprintf('  Best Performance: %.2f%% (seed %d)\n', max(results_flf(:,2))*100, best_seed);
fprintf('  Coefficient of Variation: %.2f%% (accuracy)\n', (std(results_flf(:,2))/mean(results_flf(:,2)))*100);

fprintf('\nReliability Assessment:\n');
if (std(results_flf(:,2))/mean(results_flf(:,2)))*100 < 2.0
    fprintf('  EXCELLENT: Very stable performance across seeds\n');
elseif (std(results_flf(:,2))/mean(results_flf(:,2)))*100 < 5.0
    fprintf('  GOOD: Stable performance with minor variations\n');
else
    fprintf('  MODERATE: Some performance variation across seeds\n');
end

fprintf('\nRecommended Configuration:\n');
fprintf('  Use seed: %d for best results\n', best_seed);
fprintf('  Alpha scales: [%s]\n', num2str(alpha_scales));
fprintf('  Expected accuracy: %.2f%% ± %.2f%%\n', mean(results_flf(:,2))*100, std(results_flf(:,2))*100);

%% ====== HELPER FUNCTIONS ======

function [accuracy, precision, recall, f1, cm] = calculate_metrics(y_true, y_pred)
    cm = confusionmat(y_true, y_pred);
    
    TP = cm(2,2);
    TN = cm(1,1);
    FP = cm(1,2);
    FN = cm(2,1);
    
    accuracy = (TP + TN) / sum(cm(:));
    
    if (TP + FP) > 0
        precision = TP / (TP + FP);
    else
        precision = 0;
    end
    
    if (TP + FN) > 0
        recall = TP / (TP + FN);
    else
        recall = 0;
    end
    
    if (precision + recall) > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;
    end
end

function [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size)
    % Extract multi-scale FLF features
    
    gun_files = dir(fullfile(gun_folder, '*.jpg'));
    gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];
    [~, idx] = sort({gun_files.name});
    gun_files = gun_files(idx);
    
    no_gun_files = dir(fullfile(no_gun_folder, '*.jpg'));
    no_gun_files = [no_gun_files; dir(fullfile(no_gun_folder, '*.png'))];
    [~, idx] = sort({no_gun_files.name});
    no_gun_files = no_gun_files(idx);
    
    total_images = length(gun_files) + length(no_gun_files);
    n_features = n * n * length(alpha_scales);
    features = zeros(total_images, n_features);
    labels = zeros(total_images, 1);
    
    idx = 1;
    
    % Process gun images
    fprintf('Processing gun images (FLF): ');
    for i = 1:length(gun_files)
        img_path = fullfile(gun_folder, gun_files(i).name);
        img = load_and_preprocess(img_path, image_size);
        features(idx, :) = extract_FLF_features(img, n, alpha_scales);
        labels(idx) = 1;
        idx = idx + 1;
        if mod(i, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' Done\n');
    
    % Process no-gun images
    fprintf('Processing no-gun images (FLF): ');
    for i = 1:length(no_gun_files)
        img_path = fullfile(no_gun_folder, no_gun_files(i).name);
        img = load_and_preprocess(img_path, image_size);
        features(idx, :) = extract_FLF_features(img, n, alpha_scales);
        labels(idx) = 0;
        idx = idx + 1;
        if mod(i, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' Done\n');
    
    features = features(1:idx-1, :);
    labels = labels(1:idx-1);
    features = normalize(features, 'range');
end

function [features, labels] = extract_features_single_scale(gun_folder, no_gun_folder, n, alpha, image_size)
    % Extract single-scale FLF features for individual scale analysis
    [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha, image_size);
end

function features = extract_FLF_features(img, n, alpha_scales)
    features = [];
    for alpha = alpha_scales
        C = numeric_fractional_legendre_coef_matrix_2D(img, n, alpha, alpha);
        features = [features, C(:)'];
    end
end

function img = load_and_preprocess(img_path, target_size)
    img = imread(img_path);
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    img = imresize(img, target_size);
    img = double(img) / 255.0;
end