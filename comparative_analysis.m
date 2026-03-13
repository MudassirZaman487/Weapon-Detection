%% Comprehensive Comparison: Fractional Legendre vs HOG vs SURF
% Complete comparative analysis for gun detection using three feature extraction methods
% Metrics: Accuracy, F1-Score, Precision, Recall, Time per image, Memory per image, Features

clear; clc; close all;

%% SET RANDOM SEED FOR REPRODUCIBILITY
rng(21);  % Fixed seed for all methods

%% Common Parameters
image_size = [64, 64];
train_ratio = 0.8;
gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

% Get total number of images for calculations
gun_files = [dir(fullfile(gun_folder, '*.jpg')); dir(fullfile(gun_folder, '*.png'))];
no_gun_files = [dir(fullfile(no_gun_folder, '*.jpg')); dir(fullfile(no_gun_folder, '*.png'))];
total_images = length(gun_files) + length(no_gun_files);

fprintf('=================================================================\n');
fprintf('COMPREHENSIVE COMPARISON: FLF vs HOG vs SURF for Gun Detection\n');
fprintf('=================================================================\n');
fprintf('Total Images: %d (Gun: %d, No-Gun: %d)\n', total_images, length(gun_files), length(no_gun_files));
fprintf('Image Size: [%d, %d]\n', image_size(1), image_size(2));
fprintf('Train/Test Split: %.0f%%/%.0f%%\n\n', train_ratio*100, (1-train_ratio)*100);

%% Initialize Results Structure
results = struct();
methods = {'FLF', 'HOG', 'SURF'};

%% ========================================================================
%% METHOD 1: FRACTIONAL LEGENDRE FUNCTIONS (FLF)
%% ========================================================================
fprintf('1. FRACTIONAL LEGENDRE FUNCTIONS (FLF)\n');
fprintf('======================================\n');

% FLF Parameters
n = 10;  % Polynomial order
alpha_scales = [0.5, 1, 1.5];  % Multi-scale fractional orders
flf_features = n * n * length(alpha_scales);  % 10*10*3 = 300

fprintf('Parameters:\n');
fprintf('  Polynomial order: %d × %d = %d per scale\n', n, n, n*n);
fprintf('  Alpha scales: [%s]\n', num2str(alpha_scales));
fprintf('  Total features: %d\n\n', flf_features);

% Extract FLF features
fprintf('Extracting FLF features...\n');
tic;
[features_flf, labels_flf] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);
flf_extraction_time = toc;

% Split data
[X_train_flf, X_test_flf, y_train_flf, y_test_flf, ~, ~] = split_data_fixed(features_flf, labels_flf, train_ratio);

% Train SVM
fprintf('Training FLF-SVM...\n');
tic;
svm_flf = train_svm_fixed(X_train_flf, y_train_flf);
flf_training_time = toc;

% Test and evaluate
fprintf('Evaluating FLF...\n');
tic;
[acc_flf, prec_flf, rec_flf, f1_flf, cm_flf] = evaluate_classifier(svm_flf, X_test_flf, y_test_flf);
flf_testing_time = toc;

% Store results
results.FLF.accuracy = acc_flf;
results.FLF.precision = prec_flf;
results.FLF.recall = rec_flf;
results.FLF.f1_score = f1_flf;
results.FLF.num_features = flf_features;
results.FLF.extraction_time = flf_extraction_time;
results.FLF.training_time = flf_training_time;
results.FLF.testing_time = flf_testing_time;
results.FLF.time_per_image = flf_extraction_time / total_images;
results.FLF.memory_per_image = estimate_memory_usage(flf_features, 1);
results.FLF.confusion_matrix = cm_flf;

fprintf('FLF Results: Acc=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f\n\n', acc_flf, f1_flf, prec_flf, rec_flf);

%% ========================================================================
%% METHOD 2: HISTOGRAM OF ORIENTED GRADIENTS (HOG)
%% ========================================================================
fprintf('2. HISTOGRAM OF ORIENTED GRADIENTS (HOG)\n');
fprintf('========================================\n');

% HOG Parameters
cell_size = [8, 8];
block_size = [2, 2];
num_bins = 9;
block_overlap = [1, 1];

fprintf('Parameters:\n');
fprintf('  Cell Size: [%d, %d]\n', cell_size(1), cell_size(2));
fprintf('  Block Size: [%d, %d] cells\n', block_size(1), block_size(2));
fprintf('  Orientation Bins: %d\n', num_bins);
fprintf('  Block Overlap: [%d, %d] cells\n\n', block_overlap(1), block_overlap(2));

% Extract HOG features
fprintf('Extracting HOG features...\n');
tic;
[features_hog, labels_hog, hog_features] = extract_all_HOG_features(gun_folder, no_gun_folder, ...
    image_size, cell_size, block_size, num_bins, block_overlap);
hog_extraction_time = toc;

% Split data (same indices as FLF for fair comparison)
[X_train_hog, X_test_hog, y_train_hog, y_test_hog, ~, ~] = split_data_fixed(features_hog, labels_hog, train_ratio);

% Train SVM
fprintf('Training HOG-SVM...\n');
tic;
svm_hog = train_svm_fixed(X_train_hog, y_train_hog);
hog_training_time = toc;

% Test and evaluate
fprintf('Evaluating HOG...\n');
tic;
[acc_hog, prec_hog, rec_hog, f1_hog, cm_hog] = evaluate_classifier(svm_hog, X_test_hog, y_test_hog);
hog_testing_time = toc;

% Store results
results.HOG.accuracy = acc_hog;
results.HOG.precision = prec_hog;
results.HOG.recall = rec_hog;
results.HOG.f1_score = f1_hog;
results.HOG.num_features = hog_features;
results.HOG.extraction_time = hog_extraction_time;
results.HOG.training_time = hog_training_time;
results.HOG.testing_time = hog_testing_time;
results.HOG.time_per_image = hog_extraction_time / total_images;
results.HOG.memory_per_image = estimate_memory_usage(hog_features, 1);
results.HOG.confusion_matrix = cm_hog;

fprintf('HOG Results: Acc=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f\n\n', acc_hog, f1_hog, prec_hog, rec_hog);

%% ========================================================================
%% METHOD 3: SPEEDED-UP ROBUST FEATURES (SURF)
%% ========================================================================
fprintf('3. SPEEDED-UP ROBUST FEATURES (SURF)\n');
fprintf('====================================\n');

% SURF Parameters
surf_threshold = 500;
num_features_per_image = 50;
vocab_size = 500;  % Bag of Features vocabulary size

fprintf('Parameters:\n');
fprintf('  Detection Threshold: %d\n', surf_threshold);
fprintf('  Features per image: %d (max)\n', num_features_per_image);
fprintf('  Vocabulary Size: %d (Bag of Features)\n\n', vocab_size);

% Extract SURF features
fprintf('Extracting SURF features...\n');
tic;
[features_surf, labels_surf, surf_features, avg_keypoints] = extract_all_SURF_features(gun_folder, no_gun_folder, ...
    image_size, surf_threshold, num_features_per_image);
surf_extraction_time = toc;

% Split data (same indices as others for fair comparison)
[X_train_surf, X_test_surf, y_train_surf, y_test_surf, ~, ~] = split_data_fixed(features_surf, labels_surf, train_ratio);

% Train SVM
fprintf('Training SURF-SVM...\n');
tic;
svm_surf = train_svm_fixed(X_train_surf, y_train_surf);
surf_training_time = toc;

% Test and evaluate
fprintf('Evaluating SURF...\n');
tic;
[acc_surf, prec_surf, rec_surf, f1_surf, cm_surf] = evaluate_classifier(svm_surf, X_test_surf, y_test_surf);
surf_testing_time = toc;

% Store results
results.SURF.accuracy = acc_surf;
results.SURF.precision = prec_surf;
results.SURF.recall = rec_surf;
results.SURF.f1_score = f1_surf;
results.SURF.num_features = surf_features;
results.SURF.extraction_time = surf_extraction_time;
results.SURF.training_time = surf_training_time;
results.SURF.testing_time = surf_testing_time;
results.SURF.time_per_image = surf_extraction_time / total_images;
results.SURF.memory_per_image = estimate_memory_usage(surf_features, 1);
results.SURF.confusion_matrix = cm_surf;
results.SURF.avg_keypoints = avg_keypoints;

fprintf('SURF Results: Acc=%.4f, F1=%.4f, Prec=%.4f, Rec=%.4f\n\n', acc_surf, f1_surf, prec_surf, rec_surf);

%% ========================================================================
%% COMPREHENSIVE COMPARISON TABLE
%% ========================================================================
fprintf('\n=================================================================\n');
fprintf('                    COMPREHENSIVE COMPARISON TABLE\n');
fprintf('=================================================================\n');
fprintf('Method    | Features | Accuracy | Precision | Recall | F1-Score | Time/img | Memory/img\n');
fprintf('----------|----------|----------|-----------|--------|----------|----------|----------\n');
fprintf('FLF       |   %4d   |  %.4f  |   %.4f  | %.4f |  %.4f  | %.2e | %.2f KB\n', ...
    results.FLF.num_features, results.FLF.accuracy, results.FLF.precision, ...
    results.FLF.recall, results.FLF.f1_score, results.FLF.time_per_image, results.FLF.memory_per_image*1024);
fprintf('HOG       |   %4d   |  %.4f  |   %.4f  | %.4f |  %.4f  | %.2e | %.2f KB\n', ...
    results.HOG.num_features, results.HOG.accuracy, results.HOG.precision, ...
    results.HOG.recall, results.HOG.f1_score, results.HOG.time_per_image, results.HOG.memory_per_image*1024);
fprintf('SURF      |   %4d   |  %.4f  |   %.4f  | %.4f |  %.4f  | %.2e | %.2f KB\n', ...
    results.SURF.num_features, results.SURF.accuracy, results.SURF.precision, ...
    results.SURF.recall, results.SURF.f1_score, results.SURF.time_per_image, results.SURF.memory_per_image*1024);

%% Ranking Analysis
fprintf('\n=== RANKING ANALYSIS ===\n');
accuracies = [results.FLF.accuracy, results.HOG.accuracy, results.SURF.accuracy];
f1_scores = [results.FLF.f1_score, results.HOG.f1_score, results.SURF.f1_score];
times = [results.FLF.time_per_image, results.HOG.time_per_image, results.SURF.time_per_image];
features = [results.FLF.num_features, results.HOG.num_features, results.SURF.num_features];

[~, acc_rank] = sort(accuracies, 'descend');
[~, f1_rank] = sort(f1_scores, 'descend');
[~, time_rank] = sort(times, 'ascend');  % Ascending for speed
[~, feat_rank] = sort(features, 'ascend');  % Ascending for compactness

method_names = {'FLF', 'HOG', 'SURF'};
fprintf('Best Accuracy:    %s (%.4f)\n', method_names{acc_rank(1)}, accuracies(acc_rank(1)));
fprintf('Best F1-Score:    %s (%.4f)\n', method_names{f1_rank(1)}, f1_scores(f1_rank(1)));
fprintf('Fastest:          %s (%.2e sec/img)\n', method_names{time_rank(1)}, times(time_rank(1)));
fprintf('Most Compact:     %s (%d features)\n', method_names{feat_rank(1)}, features(feat_rank(1)));

%% Detailed Performance Analysis
fprintf('\n=== DETAILED PERFORMANCE ANALYSIS ===\n');
for i = 1:length(methods)
    method = methods{i};
    cm = results.(method).confusion_matrix;
    
    % Calculate additional metrics
    sensitivity = cm(2,2) / (cm(2,2) + cm(2,1));  % Recall
    specificity = cm(1,1) / (cm(1,1) + cm(1,2));
    fpr = cm(1,2) / (cm(1,1) + cm(1,2));  % False Positive Rate
    fnr = cm(2,1) / (cm(2,1) + cm(2,2));  % False Negative Rate
    
    fprintf('\n%s Method:\n', method);
    fprintf('  Sensitivity (True Positive Rate): %.2f%%\n', sensitivity*100);
    fprintf('  Specificity (True Negative Rate): %.2f%%\n', specificity*100);
    fprintf('  False Positive Rate: %.2f%%\n', fpr*100);
    fprintf('  False Negative Rate: %.2f%%\n', fnr*100);
    fprintf('  Feature Extraction Time: %.2f sec (%.2e sec/img)\n', ...
        results.(method).extraction_time, results.(method).time_per_image);
    fprintf('  Training Time: %.2f sec\n', results.(method).training_time);
    fprintf('  Testing Time: %.2f sec\n', results.(method).testing_time);
end

%% Statistical Significance Test (if possible)
fprintf('\n=== STATISTICAL ANALYSIS ===\n');
fprintf('Accuracy differences:\n');
fprintf('  FLF vs HOG:  %.4f (%.2f%% %s)\n', ...
    results.FLF.accuracy - results.HOG.accuracy, ...
    abs(results.FLF.accuracy - results.HOG.accuracy)*100, ...
    iif(results.FLF.accuracy > results.HOG.accuracy, 'better', 'worse'));
fprintf('  FLF vs SURF: %.4f (%.2f%% %s)\n', ...
    results.FLF.accuracy - results.SURF.accuracy, ...
    abs(results.FLF.accuracy - results.SURF.accuracy)*100, ...
    iif(results.FLF.accuracy > results.SURF.accuracy, 'better', 'worse'));
fprintf('  HOG vs SURF: %.4f (%.2f%% %s)\n', ...
    results.HOG.accuracy - results.SURF.accuracy, ...
    abs(results.HOG.accuracy - results.SURF.accuracy)*100, ...
    iif(results.HOG.accuracy > results.SURF.accuracy, 'better', 'worse'));

%% Efficiency Analysis
fprintf('\n=== EFFICIENCY ANALYSIS ===\n');
flf_efficiency = results.FLF.f1_score / results.FLF.time_per_image;
hog_efficiency = results.HOG.f1_score / results.HOG.time_per_image;
surf_efficiency = results.SURF.f1_score / results.SURF.time_per_image;

fprintf('Performance-Speed Efficiency (F1/Time):\n');
fprintf('  FLF:  %.2e\n', flf_efficiency);
fprintf('  HOG:  %.2e\n', hog_efficiency);
fprintf('  SURF: %.2e\n', surf_efficiency);

[max_eff, best_eff_idx] = max([flf_efficiency, hog_efficiency, surf_efficiency]);
fprintf('Most Efficient: %s\n', method_names{best_eff_idx});

%% Generate Visualizations
create_comparison_plots(results, methods);

%% Save comprehensive results
save('comprehensive_comparison_results.mat', 'results', 'total_images', 'image_size', 'train_ratio');
fprintf('\nComprehensive results saved to comprehensive_comparison_results.mat\n');

%% ====== VISUALIZATION FUNCTION ======
function create_comparison_plots(results, methods)
    % Create comprehensive comparison plots
    
    figure;
    
    % Subplot 1: Performance Metrics
    subplot(2, 3, 1);
    metrics = {'Accuracy', 'Precision', 'Recall', 'F1-Score'};
    flf_vals = [results.FLF.accuracy, results.FLF.precision, results.FLF.recall, results.FLF.f1_score];
    hog_vals = [results.HOG.accuracy, results.HOG.precision, results.HOG.recall, results.HOG.f1_score];
    surf_vals = [results.SURF.accuracy, results.SURF.precision, results.SURF.recall, results.SURF.f1_score];
    
    x = 1:4;
    bar(x, [flf_vals; hog_vals; surf_vals]', 'grouped');
    set(gca, 'XTickLabel', metrics);
    legend('FLF', 'HOG', 'SURF', 'Location', 'best');
    title('Performance Metrics Comparison');
    ylabel('Score');
    grid on;
    
    % Subplot 2: Feature Dimensions
    subplot(2, 3, 2);
    features = [results.FLF.num_features, results.HOG.num_features, results.SURF.num_features];
    bar(features);
    set(gca, 'XTickLabel', methods);
    title('Number of Features');
    ylabel('Feature Count');
    grid on;
    
    % Subplot 3: Time per Image
    subplot(2, 3, 3);
    times = [results.FLF.time_per_image, results.HOG.time_per_image, results.SURF.time_per_image];
    bar(times);
    set(gca, 'XTickLabel', methods);
    title('Processing Time per Image');
    ylabel('Time (seconds)');
    grid on;
    
    % Subplot 4: Memory Usage
    subplot(2, 3, 4);
    memory = [results.FLF.memory_per_image, results.HOG.memory_per_image, results.SURF.memory_per_image];
    bar(memory);
    set(gca, 'XTickLabel', methods);
    title('Memory per Image');
    ylabel('Memory (MB)');
    grid on;
    
    % Subplot 5: Confusion Matrices
    subplot(2, 3, 5);
    flf_cm = results.FLF.confusion_matrix;
    confusionchart(flf_cm, {'No Gun', 'Gun'});
    title(sprintf('FLF Confusion Matrix\nAcc: %.2f%%', results.FLF.accuracy*100));
    
    % Subplot 6: ROC-like comparison
    subplot(2, 3, 6);
    precisions = [results.FLF.precision, results.HOG.precision, results.SURF.precision];
    recalls = [results.FLF.recall, results.HOG.recall, results.SURF.recall];
    scatter(recalls, precisions, 100, 'filled');
    text(recalls + 0.01, precisions, methods, 'FontSize', 10);
    xlabel('Recall');
    ylabel('Precision');
    title('Precision-Recall Comparison');
    grid on;
    axis([0 1 0 1]);
    
    sgtitle('Comprehensive Feature Method Comparison for Gun Detection');
end

%% Helper function for conditional text
function result = iif(condition, true_val, false_val)
    if condition
        result = true_val;
    else
        result = false_val;
    end
end

%% Memory usage estimation function
function memory_usage = estimate_memory_usage(feature_dim, num_samples)
    % Estimate memory usage in MB
    bytes_per_sample = feature_dim * 8;  % 8 bytes per double
    total_bytes = bytes_per_sample * num_samples;
    memory_usage = total_bytes / (1024 * 1024);  % Convert to MB
end