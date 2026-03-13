%% main_gun_detection_FLF.m
% Gun detection with Fractional-order Legendre Functions (FLF)
% Multi-scale feature extraction for improved weapon detection
% Based on: "Fractional-order Legendre functions for solving fractional-order differential equations"

 clc; close all;

%% SET RANDOM SEED FOR REPRODUCIBILITY
rng(41);  % Fixed seed - same results every time!

%% Parameters
n = 10;  % Number of Legendre polynomials per scale
alpha_scales = [0.5,1, 1.5];  % Multi-scale fractional orders
image_size = [64, 64];  % Standard image size
train_ratio = 0.8;  % 80% for training, 20% for testing

% Feature vector size: n*n*length(alpha_scales) = 10*10*3 = 300
fprintf('Fractional Legendre Features for Weapon Detection\n');
fprintf('=================================================\n');
fprintf('Polynomial order: %d × %d = %d features per scale\n', n, n, n*n);
fprintf('Alpha scales: [%s]\n', num2str(alpha_scales));
fprintf('Total features: %d\n\n', n*n*length(alpha_scales));

%% Step 1: Set data paths
gun_folder = 'D:\Research Work\Adnan bhai Research Work\Data set\Final File\Final File\dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\Data set\Final File\Final File\dataset\Negative';

fprintf('Gun Detection using Fractional Legendre Polynomials - SVM\n');
fprintf('=========================================================\n');
fprintf('Random seed: 50 (Results will be consistent across runs)\n');
fprintf('Multi-scale analysis with α = [%s]\n\n', num2str(alpha_scales));

%% Step 2: Load and process images with FLF features
fprintf('Loading images and extracting FLF features...\n');
tic;
[features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);
feature_extraction_time = toc;

fprintf('Total samples: %d\n', length(labels));
fprintf('Gun samples: %d\n', sum(labels == 1));
fprintf('No-gun samples: %d\n', sum(labels == 0));
fprintf('Feature extraction time: %.2f seconds\n', feature_extraction_time);
fprintf('Feature vector size: %d\n\n', size(features, 2));

%% Step 3: Split data into train and test (FIXED SPLIT)
[X_train, X_test, y_train, y_test, train_idx, test_idx] = split_data_fixed(features, labels, train_ratio);

fprintf('Training samples: %d\n', size(X_train, 1));
fprintf('Testing samples: %d\n\n', size(X_test, 1));

%% Step 4: Train SVM with hyperparameter optimization
fprintf('Training SVM classifier with FLF features...\n');
tic;

% Train with hyperparameter optimization for best results
svm_model = train_svm_optimized_FLF(X_train, y_train);
training_time = toc;

%% Step 5: Test and evaluate
fprintf('Evaluating classifier...\n');
[accuracy, precision, recall, f1, cm] = evaluate_classifier(svm_model, X_test, y_test);

%% Step 6: Display results
fprintf('\n=== FRACTIONAL LEGENDRE SVM RESULTS ===\n');
fprintf('Accuracy:  %.4f (%.2f%%)\n', accuracy, accuracy*100);
fprintf('Precision: %.4f (%.2f%%)\n', precision, precision*100);
fprintf('Recall:    %.4f (%.2f%%)\n', recall, recall*100);
fprintf('F1-Score:  %.4f\n', f1);
fprintf('Training Time: %.2f seconds\n', training_time);

fprintf('\nConfusion Matrix:\n');
fprintf('               Predicted\n');
fprintf('              No Gun  Gun\n');
fprintf('Actual No Gun  %4d  %4d\n', cm(1,1), cm(1,2));
fprintf('       Gun     %4d  %4d\n', cm(2,1), cm(2,2));

% Calculate detailed metrics
fprintf('\nDetailed Performance Metrics:\n');
fprintf('True Positive Rate (Sensitivity): %.2f%%\n', recall*100);
fprintf('True Negative Rate (Specificity): %.2f%%\n', cm(1,1)/(cm(1,1)+cm(1,2))*100);
fprintf('False Positive Rate: %.2f%%\n', cm(1,2)/(cm(1,1)+cm(1,2))*100);
fprintf('False Negative Rate: %.2f%%\n', cm(2,1)/(cm(2,1)+cm(2,2))*100);

%% Step 7: Save model with all parameters
save('gun_detection_model_FLF.mat', 'svm_model', 'n', 'alpha_scales', 'image_size', ...
     'train_idx', 'test_idx', 'accuracy', 'precision', 'recall', 'f1', ...
     'feature_extraction_time', 'training_time');
fprintf('\nModel saved to gun_detection_model_FLF.mat\n');

%% Plot confusion matrix
figure('Position', [100, 100, 600, 500]);
confusionchart(cm, {'No Gun', 'Gun'});
title(sprintf('FLF-SVM Confusion Matrix - Accuracy: %.2f%%', accuracy*100));

%% Analyze feature importance across scales
fprintf('\n=== Feature Analysis Across Scales ===\n');
analyze_feature_importance_FLF(X_train, y_train, n, alpha_scales);

%% ====== HELPER FUNCTIONS ======



