%% main_gun_detection_FLF_Legendre_Kernel.m
% Gun detection with Fractional-order Legendre Functions (FLF)
% Using Legendre Kernel SVM instead of RBF kernel
% Multi-scale feature extraction for improved weapon detection

clc; clear; close all;

%% SET RANDOM SEED FOR REPRODUCIBILITY
rng(21);  % Fixed seed - same results every time!

%% Parameters
n = 10;  % Number of Legendre polynomials per scale
alpha_scales = [0.5, 1, 1.5];  % Multi-scale fractional orders
image_size = [64, 64];  % Standard image size
train_ratio = 0.8;  % 80% for training, 20% for testing
legendre_order = 5;  % Order for Legendre kernel (can be different from feature extraction order)

% Feature vector size: n*n*length(alpha_scales) = 10*10*3 = 300
fprintf('Fractional Legendre Features with Legendre Kernel SVM\n');
fprintf('=====================================================\n');
fprintf('Feature extraction polynomial order: %d × %d = %d features per scale\n', n, n, n*n);
fprintf('Legendre kernel order: %d\n', legendre_order);
fprintf('Alpha scales: [%s]\n', num2str(alpha_scales));
fprintf('Total features: %d\n\n', n*n*length(alpha_scales));

%% Step 1: Set data paths
gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

fprintf('Gun Detection using Fractional Legendre Polynomials - Legendre Kernel SVM\n');
fprintf('========================================================================\n');
fprintf('Random seed: 21 (Results will be consistent across runs)\n');
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

%% Step 4: Train SVM with Legendre kernel
fprintf('Training SVM classifier with Legendre kernel...\n');
tic;

% Normalize features to [-1, 1] for Legendre polynomial domain
X_train_norm = normalize_to_legendre_domain(X_train);
X_test_norm = normalize_to_legendre_domain(X_test);

% Create custom Legendre kernel function
legendre_kernel_func = @(X1, X2) compute_legendre_kernel_matrix(X1, X2, legendre_order);

% Train SVM with custom Legendre kernel
svm_model = fitcsvm(X_train_norm, y_train, ...
    'KernelFunction', 'mysigma', ...
    'KernelScale', 1, ...
    'BoxConstraint', 1, ...
    'Standardize', false, ...
    'ClassNames', [0, 1]);

% Store the kernel function and normalization parameters in the model
svm_model.KernelFunction = legendre_kernel_func;
svm_model.LegendreOrder = legendre_order;
svm_model.NormParams = struct('min_vals', min(X_train), 'max_vals', max(X_train));

training_time = toc;

%% Step 5: Test and evaluate
fprintf('Evaluating classifier...\n');

% Predict using the custom kernel
scores = zeros(size(X_test_norm, 1), 1);
for i = 1:size(X_test_norm, 1)
    K = compute_legendre_kernel_matrix(X_test_norm(i,:), X_train_norm, legendre_order);
    scores(i) = K * svm_model.Alpha .* (2*svm_model.IsSupportVector - 1) + svm_model.Bias;
end
y_pred = double(scores > 0);

% Calculate metrics
accuracy = sum(y_pred == y_test) / length(y_test);
cm = confusionmat(y_test, y_pred);
precision = cm(2,2) / (cm(2,2) + cm(1,2));
recall = cm(2,2) / (cm(2,2) + cm(2,1));
f1 = 2 * (precision * recall) / (precision + recall);

%% Step 6: Display results
fprintf('\n=== FRACTIONAL LEGENDRE with LEGENDRE KERNEL SVM RESULTS ===\n');
fprintf('Legendre Kernel Order: %d\n', legendre_order);
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
save('gun_detection_model_FLF_Legendre_Kernel.mat', 'svm_model', 'n', 'alpha_scales', ...
     'image_size', 'legendre_order', 'train_idx', 'test_idx', 'accuracy', ...
     'precision', 'recall', 'f1', 'feature_extraction_time', 'training_time');
fprintf('\nModel saved to gun_detection_model_FLF_Legendre_Kernel.mat\n');

%% Plot confusion matrix
figure('Position', [100, 100, 600, 500]);
confusionchart(cm, {'No Gun', 'Gun'});
title(sprintf('FLF-Legendre Kernel SVM - Accuracy: %.2f%%', accuracy*100));

%% Analyze feature importance across scales
fprintf('\n=== Feature Analysis Across Scales ===\n');
analyze_feature_importance_FLF(X_train, y_train, n, alpha_scales);

%% Compare with RBF kernel
fprintf('\n=== Comparison with RBF Kernel ===\n');
% Train RBF SVM for comparison
tic;
svm_rbf = fitcsvm(X_train, y_train, ...
    'KernelFunction', 'rbf', ...
    'KernelScale', 'auto', ...
    'BoxConstraint', 1, ...
    'Standardize', true, ...
    'ClassNames', [0, 1]);
rbf_time = toc;

[y_pred_rbf, ~] = predict(svm_rbf, X_test);
accuracy_rbf = sum(y_pred_rbf == y_test) / length(y_test);
fprintf('RBF Kernel Accuracy: %.2f%% (Training time: %.2f s)\n', accuracy_rbf*100, rbf_time);
fprintf('Legendre Kernel Accuracy: %.2f%% (Training time: %.2f s)\n', accuracy*100, training_time);
fprintf('Improvement: %.2f%%\n', (accuracy - accuracy_rbf)*100);

%% ====== HELPER FUNCTIONS ======

function X_norm = normalize_to_legendre_domain(X)
    % Normalize features to [-1, 1] for Legendre polynomial domain
    X_norm = zeros(size(X));
    for i = 1:size(X, 2)
        min_val = min(X(:,i));
        max_val = max(X(:,i));
        if max_val > min_val
            X_norm(:,i) = 2 * (X(:,i) - min_val) / (max_val - min_val) - 1;
        else
            X_norm(:,i) = 0;
        end
    end
end

function K = compute_legendre_kernel_matrix(X1, X2, order)
    % Compute Legendre kernel matrix between X1 and X2
    % X1: n1 x d matrix
    % X2: n2 x d matrix
    % order: maximum order of Legendre polynomials
    % Returns: n1 x n2 kernel matrix
    
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    d = size(X1, 2);
    
    K = zeros(n1, n2);
    
    for i = 1:n1
        for j = 1:n2
            % Compute kernel value between X1(i,:) and X2(j,:)
            kernel_val = 0;
            for dim = 1:d
                % Sum of products of Legendre polynomials
                for p = 0:order
                    P1 = legendre_polynomial(p, X1(i, dim));
                    P2 = legendre_polynomial(p, X2(j, dim));
                    kernel_val = kernel_val + P1 * P2;
                end
            end
            K(i, j) = kernel_val / d;  % Normalize by dimension
        end
    end
end

function P = legendre_polynomial(n, x)
    % Compute Legendre polynomial of order n at point x
    % Using recurrence relation for efficiency
    
    if n == 0
        P = 1;
    elseif n == 1
        P = x;
    else
        P0 = 1;
        P1 = x;
        for k = 2:n
            P = ((2*k-1)*x*P1 - (k-1)*P0) / k;
            P0 = P1;
            P1 = P;
        end
    end
end

%% Alternative: Fractional Legendre Kernel (Advanced)
function K = compute_fractional_legendre_kernel_matrix(X1, X2, order, alpha)
    % Compute Fractional Legendre kernel matrix
    % Incorporates fractional calculus into the kernel
    
    n1 = size(X1, 1);
    n2 = size(X2, 1);
    d = size(X1, 2);
    
    K = zeros(n1, n2);
    
    for i = 1:n1
        for j = 1:n2
            kernel_val = 0;
            for dim = 1:d
                % Sum of products of fractional Legendre functions
                for p = 0:order
                    % Fractional Legendre function evaluation
                    FL1 = fractional_legendre_function(p, alpha, X1(i, dim));
                    FL2 = fractional_legendre_function(p, alpha, X2(j, dim));
                    kernel_val = kernel_val + FL1 * FL2;
                end
            end
            K(i, j) = kernel_val / d;
        end
    end
end

function FL = fractional_legendre_function(n, alpha, x)
    % Simplified fractional Legendre function
    % This is a placeholder - implement your specific fractional Legendre formula
    
    % Standard Legendre polynomial
    P = legendre_polynomial(n, x);
    
    % Apply fractional modification (simplified)
    % You should replace this with your actual fractional Legendre implementation
    FL = P * (1 + alpha * abs(x)^alpha) / (1 + alpha);
end