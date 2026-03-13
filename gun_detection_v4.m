%% gun_detection_video_exact_normalization.m
% Video detection using exact same normalization as training

clear; clc; close all;

%% Load model
load('gun_detection_model_FLF.mat');

%% First, save normalization parameters from training data
fprintf('Computing normalization parameters from training data...\n');
gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

% Load training features to get exact normalization
[all_features, ~] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);

% IMPORTANT: normalize() does per-feature normalization
% Get min and max for each feature dimension
feature_mins = min(all_features, [], 1);  % 1x300 vector
feature_maxs = max(all_features, [], 1);  % 1x300 vector
feature_ranges = feature_maxs - feature_mins;

% Handle zero ranges
feature_ranges(feature_ranges == 0) = 1;

fprintf('Normalization parameters computed.\n');
clear all_features;  % Free memory

%% Process video
videoObject = VideoReader('voilancevideo.mp4');
detector = vision.CascadeObjectDetector('Libraries/gunDetectionUsingHog.xml');

outputVideo = VideoWriter('gun_detection_exact_norm.mp4');
outputVideo.FrameRate = videoObject.FrameRate;
open(outputVideo);

figure('Position', [100, 100, 800, 600]);

guns_detected = 0;
false_positives = 0;

%% Process frames
for frame = 1:min(6231, videoObject.NumberOfFrames)  % First 500 frames
    img = read(videoObject, frame);
    
    % Get detections
    bbox = step(detector, img);
    
    verified_bbox = [];
    
    if ~isempty(bbox)
        fprintf('Frame %d: %d initial detections\n', frame, size(bbox, 1));
        
        for i = 1:size(bbox, 1)
            % Extract region
            region = imcrop(img, bbox(i,:));
            
            % Preprocess exactly as training
            if size(region, 3) == 3
                region = rgb2gray(region);
            end
            region = imresize(region, image_size);
            region = double(region) / 255.0;
            
            % Extract FLF features
            features = [];
            for alpha = alpha_scales
                C = numeric_fractional_legendre_coef_matrix_2D(region, n, alpha, alpha);
                features = [features, C(:)'];
            end
            
            % Apply EXACT same normalization as training
            % normalize(X, 'range') does: (X - min) ./ (max - min) per column
            features_normalized = (features - feature_mins) ./ feature_ranges;
            
            % Ensure in [0,1] range
            features_normalized = max(0, min(1, features_normalized));
            
            % Predict
            prediction = predict(svm_model, features_normalized);
            
            if prediction == 1
                verified_bbox = [verified_bbox; bbox(i,:)];
                guns_detected = guns_detected + 1;
                fprintf('  Detection %d: GUN VERIFIED\n', i);
            else
                false_positives = false_positives + 1;
                fprintf('  Detection %d: False positive\n', i);
            end
        end
    end
    
    % Annotate frame
    annotated = img;
    
    % Show initial detections in yellow
    if ~isempty(bbox)
        for i = 1:size(bbox, 1)
            annotated = insertShape(annotated, 'Rectangle', bbox(i,:), ...
                'Color', 'yellow', 'LineWidth', 2);
        end
    end
    
    % Show verified guns in red
    if ~isempty(verified_bbox)
        annotated = insertShape(annotated, 'Rectangle', verified_bbox, ...
            'Color', 'red', 'LineWidth', 3);
        for i = 1:size(verified_bbox, 1)
            annotated = insertText(annotated, verified_bbox(i,1:2), 'GUN', ...
                'FontSize', 16, 'BoxColor', 'red', 'TextColor', 'white');
        end
    end
    
    % Add stats
    stats_text = sprintf('Frame %d | Guns: %d | FP filtered: %d', ...
        frame, size(verified_bbox, 1), false_positives);
    annotated = insertText(annotated, [10 10], stats_text, ...
        'FontSize', 14, 'BoxColor', 'black', 'TextColor', 'white');
    
    % Display and save
    imshow(annotated);
    title('Gun Detection (Yellow=Initial, Red=Verified)');
    drawnow;
    
    writeVideo(outputVideo, annotated);
end

close(outputVideo);

%% Summary
fprintf('\n=== DETECTION SUMMARY ===\n');
fprintf('Total guns detected: %d\n', guns_detected);
fprintf('Total false positives filtered: %d\n', false_positives);
fprintf('Detection rate: %.2f%%\n', guns_detected/(guns_detected+false_positives)*100);

%% Save normalization parameters for future use
save('FLF_normalization_params.mat', 'feature_mins', 'feature_maxs', 'feature_ranges');
fprintf('\nNormalization parameters saved for future use.\n');

%% Helper function
function [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size)
    gun_files = dir(fullfile(gun_folder, '*.jpg'));
    gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];
    no_gun_files = dir(fullfile(no_gun_folder, '*.jpg'));
    no_gun_files = [no_gun_files; dir(fullfile(no_gun_folder, '*.png'))];
    
    total_images = length(gun_files) + length(no_gun_files);
    n_features = n * n * length(alpha_scales);
    features = zeros(total_images, n_features);
    labels = zeros(total_images, 1);
    
    idx = 1;
    
    % Gun images
    for i = 1:length(gun_files)
        img_path = fullfile(gun_folder, gun_files(i).name);
        img = imread(img_path);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        img = imresize(img, image_size);
        img = double(img) / 255.0;
        
        feat = [];
        for alpha = alpha_scales
            C = numeric_fractional_legendre_coef_matrix_2D(img, n, alpha, alpha);
            feat = [feat, C(:)'];
        end
        features(idx, :) = feat;
        labels(idx) = 1;
        idx = idx + 1;
    end
    
    % No-gun images
    for i = 1:length(no_gun_files)
        img_path = fullfile(no_gun_folder, no_gun_files(i).name);
        img = imread(img_path);
        if size(img, 3) == 3
            img = rgb2gray(img);
        end
        img = imresize(img, image_size);
        img = double(img) / 255.0;
        
        feat = [];
        for alpha = alpha_scales
            C = numeric_fractional_legendre_coef_matrix_2D(img, n, alpha, alpha);
            feat = [feat, C(:)'];
        end
        features(idx, :) = feat;
        labels(idx) = 0;
        idx = idx + 1;
    end
    
    features = features(1:idx-1, :);
    labels = labels(1:idx-1);
end