%% save_FLF_model_with_normalization.m
% Re-save the FLF model with normalization parameters

clear; clc;

%% Load existing model
load('gun_detection_model_FLF.mat');

%% Load training data to compute normalization parameters
fprintf('Computing normalization parameters from training data...\n');

gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

% Extract all training features
[features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);

% Compute normalization parameters BEFORE normalize()
feature_min_raw = min(features(:));
feature_max_raw = max(features(:));
feature_mean_raw = mean(features(:));
feature_std_raw = std(features(:));

% Also compute per-feature statistics
feature_min_vec = min(features, [], 1);
feature_max_vec = max(features, [], 1);
feature_mean_vec = mean(features, 1);
feature_std_vec = std(features, 0, 1);

% Apply normalization (as was done in training)
features_normalized = normalize(features, 'range');

fprintf('Raw feature range: [%.4f, %.4f]\n', feature_min_raw, feature_max_raw);
fprintf('Feature dimensions: %d\n', size(features, 2));

%% Save enhanced model
save('gun_detection_model_FLF_enhanced.mat', ...
    'svm_model', 'n', 'alpha_scales', 'image_size', ...
    'train_idx', 'test_idx', 'accuracy', 'precision', 'recall', 'f1', ...
    'feature_min_vec', 'feature_max_vec', 'feature_mean_vec', 'feature_std_vec', ...
    'feature_min_raw', 'feature_max_raw');

fprintf('\nEnhanced model saved with normalization parameters!\n');

%% Test the normalization
% Extract a test sample
test_idx_sample = test_idx(1:5);
test_features_raw = features(test_idx_sample, :);

% Method 1: Per-feature normalization (what normalize() does)
test_normalized_1 = (test_features_raw - feature_min_vec) ./ (feature_max_vec - feature_min_vec);

% Method 2: Global normalization
test_normalized_2 = (test_features_raw - feature_min_raw) / (feature_max_raw - feature_min_raw);

% Check predictions
predictions_1 = predict(svm_model, test_normalized_1);
predictions_2 = predict(svm_model, test_normalized_2);

fprintf('\nTest normalization methods:\n');
fprintf('Method 1 (per-feature): %d correct out of 5\n', sum(predictions_1 == labels(test_idx_sample)));
fprintf('Method 2 (global): %d correct out of 5\n', sum(predictions_2 == labels(test_idx_sample)));

%% Helper function
function [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size)
    % Extract Fractional Legendre features from all images
    
    gun_files = dir(fullfile(gun_folder, '*.jpg'));
    gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];
    gun_files = sortfiles(gun_files);
    
    no_gun_files = dir(fullfile(no_gun_folder, '*.jpg'));
    no_gun_files = [no_gun_files; dir(fullfile(no_gun_folder, '*.png'))];
    no_gun_files = sortfiles(no_gun_files);
    
    total_images = length(gun_files) + length(no_gun_files);
    n_features = n * n * length(alpha_scales);
    features = zeros(total_images, n_features);
    labels = zeros(total_images, 1);
    
    idx = 1;
    
    % Process gun images
    fprintf('Processing gun images: ');
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
    fprintf('Processing no-gun images: ');
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

function sorted_files = sortfiles(files)
    [~, idx] = sort({files.name});
    sorted_files = files(idx);
end