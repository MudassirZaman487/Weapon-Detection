%% svm_FLF_seed_analysis.m
% Test Fractional Legendre Features SVM with different random seeds (20 to 60)
% Compare stability between standard and fractional Legendre approaches

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

%% Load and process images ONCE with FLF features
fprintf('Loading images and extracting FLF features...\n');
tic;
[features_flf, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);
loading_time = toc;

fprintf('Total samples: %d\n', length(labels));
fprintf('Gun samples: %d\n', sum(labels == 1));
fprintf('No-gun samples: %d\n', sum(labels == 0));
fprintf('Feature extraction time: %.2f seconds\n', loading_time);
fprintf('Feature vector size: %d\n\n', size(features_flf, 2));

%% Also extract standard Legendre features for comparison
fprintf('Extracting standard Legendre features for comparison...\n');
tic;
[features_std, ~] = extract_all_features_standard(gun_folder, no_gun_folder, n, image_size);
std_time = toc;
fprintf('Standard feature extraction time: %.2f seconds\n\n', std_time);

%% Initialize results storage
results_flf = zeros(num_seeds, 5);  % seed, accuracy, precision, recall, f1
results_std = zeros(num_seeds, 5);  % For comparison
training_times_flf = zeros(num_seeds, 1);
training_times_std = zeros(num_seeds, 1);

%% Test each seed
fprintf('Testing both methods with different seeds...\n');
fprintf('Seed | FLF Acc. | Std Acc. | FLF F1  | Std F1  | Improvement\n');
fprintf('-----|----------|----------|---------|---------|------------\n');

for i = 1:num_seeds
    seed = seed_start + i - 1;
    
    % Set random seed
    rng(seed);
    
    % Split data with current seed (same split for both methods)
    n_samples = size(features_flf, 1);
    idx = randperm(n_samples);
    n_train = round(n_samples * train_ratio);
    train_idx = idx(1:n_train);
    test_idx = idx(n_train+1:end);
    
    % === FRACTIONAL LEGENDRE ===
    X_train_flf = features_flf(train_idx, :);
    X_test_flf = features_flf(test_idx, :);
    y_train = labels(train_idx);
    y_test = labels(test_idx);
    
    % Train FLF SVM
    tic;
    svm_flf = fitcsvm(X_train_flf, y_train, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'BoxConstraint', 10, ...      % Using best parameters from optimization
        'KernelScale', 'auto');
    training_times_flf(i) = toc;
    
    % Evaluate FLF
    y_pred_flf = predict(svm_flf, X_test_flf);
    [acc_flf, prec_flf, rec_flf, f1_flf, ~] = calculate_metrics(y_test, y_pred_flf);
    results_flf(i, :) = [seed, acc_flf, prec_flf, rec_flf, f1_flf];
    
    % === STANDARD LEGENDRE ===
    X_train_std = features_std(train_idx, :);
    X_test_std = features_std(test_idx, :);
    
    % Train Standard SVM
    tic;
    svm_std = fitcsvm(X_train_std, y_train, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'BoxConstraint', 10, ...
        'KernelScale', 0.01);         % From reproduced paper
    training_times_std(i) = toc;
    
    % Evaluate Standard
    y_pred_std = predict(svm_std, X_test_std);
    [acc_std, prec_std, rec_std, f1_std, ~] = calculate_metrics(y_test, y_pred_std);
    results_std(i, :) = [seed, acc_std, prec_std, rec_std, f1_std];
    
    % Display results
    improvement = (acc_flf - acc_std) * 100;
    fprintf(' %3d |  %.4f  |  %.4f  | %.4f | %.4f | %+6.2f%%\n', ...
        seed, acc_flf, acc_std, f1_flf, f1_std, improvement);
end

%% Calculate statistics
fprintf('\n==========================================================\n');
fprintf('STATISTICAL SUMMARY - FRACTIONAL LEGENDRE\n');
fprintf('==========================================================\n\n');

% Mean and std for each metric (FLF)
metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
for i = 1:4
    metric_values_flf = results_flf(:, i+1);
    metric_values_std = results_std(:, i+1);
    
    mean_flf = mean(metric_values_flf);
    std_flf = std(metric_values_flf);
    mean_std = mean(metric_values_std);
    std_std = std(metric_values_std);
    
    fprintf('%s:\n', metrics{i});
    fprintf('  FLF:      %.4f ± %.4f (Range: %.4f to %.4f)\n', ...
        mean_flf, std_flf, min(metric_values_flf), max(metric_values_flf));
    fprintf('  Standard: %.4f ± %.4f (Range: %.4f to %.4f)\n', ...
        mean_std, std_std, min(metric_values_std), max(metric_values_std));
    fprintf('  Improvement: %+.2f%% (avg), Stability: %.1fx better\n\n', ...
        (mean_flf - mean_std)*100, std_std/std_flf);
end

%% Find best seeds
[best_f1_flf, best_idx_flf] = max(results_flf(:, 5));
best_seed_flf = results_flf(best_idx_flf, 1);

[best_f1_std, best_idx_std] = max(results_std(:, 5));
best_seed_std = results_std(best_idx_std, 1);

fprintf('==========================================================\n');
fprintf('BEST SEEDS\n');
fprintf('==========================================================\n');
fprintf('FLF Best Seed: %d (F1: %.4f)\n', best_seed_flf, best_f1_flf);
fprintf('Standard Best Seed: %d (F1: %.4f)\n', best_seed_std, best_f1_std);

%% Visualize comparative results
figure('Position', [100, 100, 1400, 800]);

% Plot 1: Accuracy comparison
subplot(2,3,1);
plot(results_flf(:,1), results_flf(:,2), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(results_std(:,1), results_std(:,2), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Random Seed');
ylabel('Accuracy');
title('Accuracy: FLF vs Standard Legendre');
legend('FLF (Multi-scale)', 'Standard (α=1)', 'Location', 'best');
grid on;
ylim([0.9, 1.0]);

% Plot 2: F1-Score comparison
subplot(2,3,2);
plot(results_flf(:,1), results_flf(:,5), 'b-o', 'LineWidth', 2, 'MarkerSize', 6);
hold on;
plot(results_std(:,1), results_std(:,5), 'r--s', 'LineWidth', 1.5, 'MarkerSize', 5);
xlabel('Random Seed');
ylabel('F1-Score');
title('F1-Score: FLF vs Standard Legendre');
legend('FLF (Multi-scale)', 'Standard (α=1)', 'Location', 'best');
grid on;
ylim([0.9, 1.0]);

% Plot 3: Box plot comparison
subplot(2,3,3);
data_compare = [results_flf(:,2), results_std(:,2), ...
                results_flf(:,5), results_std(:,5)];
boxplot(data_compare, 'Labels', {'FLF Acc.', 'Std Acc.', 'FLF F1', 'Std F1'});
ylabel('Score');
title('Performance Distribution Comparison');
grid on;

% Plot 4: Improvement over seeds
subplot(2,3,4);
improvements_acc = (results_flf(:,2) - results_std(:,2)) * 100;
improvements_f1 = (results_flf(:,5) - results_std(:,5)) * 100;
bar(results_flf(:,1), [improvements_acc, improvements_f1], 'grouped');
xlabel('Random Seed');
ylabel('Improvement (%)');
title('FLF Improvement over Standard Legendre');
legend('Accuracy', 'F1-Score', 'Location', 'best');
grid on;

% Plot 5: Stability analysis
subplot(2,3,5);
stds = [std(results_flf(:,2)), std(results_std(:,2)); ...
        std(results_flf(:,3)), std(results_std(:,3)); ...
        std(results_flf(:,4)), std(results_std(:,4)); ...
        std(results_flf(:,5)), std(results_std(:,5))];
b = bar(stds, 'grouped');
set(gca, 'XTickLabel', metrics);
ylabel('Standard Deviation');
title('Stability Comparison (Lower is Better)');
legend('FLF', 'Standard', 'Location', 'best');
grid on;

% Plot 6: Per-scale contribution
subplot(2,3,6);
% Test individual scales for best seed
rng(best_seed_flf);
idx = randperm(n_samples);
train_idx = idx(1:n_train);
test_idx = idx(n_train+1:end);

scale_accs = zeros(length(alpha_scales), 1);
for a = 1:length(alpha_scales)
    % Extract single-scale features
    [features_single, ~] = extract_features_single_scale(gun_folder, no_gun_folder, n, alpha_scales(a), image_size);
    X_train_s = features_single(train_idx, :);
    X_test_s = features_single(test_idx, :);
    
    svm_s = fitcsvm(X_train_s, y_train, 'KernelFunction', 'rbf', 'Standardize', true);
    y_pred_s = predict(svm_s, X_test_s);
    scale_accs(a) = sum(y_pred_s == y_test) / length(y_test);
end

bar(alpha_scales, scale_accs, 'FaceColor', [0.2 0.6 0.8]);
xlabel('Alpha (α)');
ylabel('Accuracy');
title(sprintf('Individual Scale Performance (Seed %d)', best_seed_flf));
ylim([0.8, 1.0]);
grid on;

sgtitle('Fractional vs Standard Legendre: Stability Analysis');

%% Save comprehensive results
save('flf_stability_analysis_results.mat', 'results_flf', 'results_std', ...
     'training_times_flf', 'training_times_std', 'alpha_scales', 'best_seed_flf');
fprintf('\nResults saved to flf_stability_analysis_results.mat\n');

%% Create detailed report
report_file = 'flf_stability_report.txt';
fid = fopen(report_file, 'w');

fprintf(fid, 'Fractional Legendre Stability Analysis Report\n');
fprintf(fid, '============================================\n\n');
fprintf(fid, 'Date: %s\n', datestr(now));
fprintf(fid, 'Seeds tested: %d to %d (%d total)\n', seed_start, seed_end, num_seeds);
fprintf(fid, 'Alpha scales: [%s]\n\n', num2str(alpha_scales));

fprintf(fid, 'AVERAGE PERFORMANCE COMPARISON:\n');
fprintf(fid, '                    FLF          Standard     Improvement\n');
fprintf(fid, 'Accuracy:    %.4f ± %.4f   %.4f ± %.4f    %+.2f%%\n', ...
    mean(results_flf(:,2)), std(results_flf(:,2)), ...
    mean(results_std(:,2)), std(results_std(:,2)), ...
    (mean(results_flf(:,2)) - mean(results_std(:,2)))*100);
fprintf(fid, 'F1-Score:    %.4f ± %.4f   %.4f ± %.4f    %+.2f%%\n\n', ...
    mean(results_flf(:,5)), std(results_flf(:,5)), ...
    mean(results_std(:,5)), std(results_std(:,5)), ...
    (mean(results_flf(:,5)) - mean(results_std(:,5)))*100);

fprintf(fid, 'STABILITY ANALYSIS:\n');
fprintf(fid, 'FLF Std Dev:      %.4f (Accuracy), %.4f (F1)\n', ...
    std(results_flf(:,2)), std(results_flf(:,5)));
fprintf(fid, 'Standard Std Dev: %.4f (Accuracy), %.4f (F1)\n', ...
    std(results_std(:,2)), std(results_std(:,5)));
fprintf(fid, 'Stability Improvement: %.1fx better\n\n', ...
    std(results_std(:,2))/std(results_flf(:,2)));

fprintf(fid, 'RECOMMENDATION:\n');
fprintf(fid, 'Use Fractional Legendre Features with α = [%s]\n', num2str(alpha_scales));
fprintf(fid, 'Best seed for reproduction: %d\n', best_seed_flf);
fprintf(fid, 'Expected performance: %.2f%% ± %.2f%% accuracy\n', ...
    mean(results_flf(:,2))*100, std(results_flf(:,2))*100);

fclose(fid);
fprintf('Detailed report saved to %s\n', report_file);

%% Final comparison summary
fprintf('\n==========================================================\n');
fprintf('FINAL COMPARISON SUMMARY\n');
fprintf('==========================================================\n');
fprintf('Fractional Legendre (Multi-scale):\n');
fprintf('  Accuracy: %.2f%% ± %.2f%%\n', mean(results_flf(:,2))*100, std(results_flf(:,2))*100);
fprintf('  F1-Score: %.4f ± %.4f\n', mean(results_flf(:,5)), std(results_flf(:,5)));
fprintf('  Best: %.2f%% (seed %d)\n\n', max(results_flf(:,2))*100, ...
    results_flf(results_flf(:,2)==max(results_flf(:,2)), 1));

fprintf('Standard Legendre (α=1):\n');
fprintf('  Accuracy: %.2f%% ± %.2f%%\n', mean(results_std(:,2))*100, std(results_std(:,2))*100);
fprintf('  F1-Score: %.4f ± %.4f\n', mean(results_std(:,5)), std(results_std(:,5)));
fprintf('  Best: %.2f%% (seed %d)\n\n', max(results_std(:,2))*100, ...
    results_std(results_std(:,2)==max(results_std(:,2)), 1));

fprintf('IMPROVEMENTS:\n');
fprintf('  Average accuracy gain: %+.2f%%\n', (mean(results_flf(:,2)) - mean(results_std(:,2)))*100);
fprintf('  Stability improvement: %.1fx\n', std(results_std(:,2))/std(results_flf(:,2)));
fprintf('  Consistency: %.1f%% of seeds show improvement\n', ...
    sum(results_flf(:,2) > results_std(:,2))/num_seeds*100);

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

function [features, labels] = extract_all_features_standard(gun_folder, no_gun_folder, n, image_size)
    % Extract standard Legendre features for comparison
    
    gun_files = dir(fullfile(gun_folder, '*.jpg'));
    gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];
    [~, idx] = sort({gun_files.name});
    gun_files = gun_files(idx);
    
    no_gun_files = dir(fullfile(no_gun_folder, '*.jpg'));
    no_gun_files = [no_gun_files; dir(fullfile(no_gun_folder, '*.png'))];
    [~, idx] = sort({no_gun_files.name});
    no_gun_files = no_gun_files(idx);
    
    total_images = length(gun_files) + length(no_gun_files);
    features = zeros(total_images, n*n);
    labels = zeros(total_images, 1);
    
    idx = 1;
    
    % Process all images
    all_files = [gun_files; no_gun_files];
    all_folders = [repmat({gun_folder}, length(gun_files), 1); ...
                   repmat({no_gun_folder}, length(no_gun_files), 1)];
    all_labels = [ones(length(gun_files), 1); zeros(length(no_gun_files), 1)];
    
    for i = 1:length(all_files)
        img_path = fullfile(all_folders{i}, all_files(i).name);
        img = load_and_preprocess(img_path, image_size);
        C = legendre_coef_matrix1_2D(img, n);
        features(i, :) = C(:)';
        labels(i) = all_labels(i);
    end
    
    features = normalize(features, 'range');
end

function [features, labels] = extract_features_single_scale(gun_folder, no_gun_folder, n, alpha, image_size)
    % Extract single-scale FLF features
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