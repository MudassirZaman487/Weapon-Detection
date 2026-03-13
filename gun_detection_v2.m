%% gun_detection_video_FLF_corrected.m
% Corrected gun detection with proper feature normalization

clear; clc; close all;

%% Load model and normalization parameters
load('gun_detection_model_FLF.mat');

% Critical: Load training data to get normalization parameters
% OR save these during training
fprintf('Loading training data for normalization parameters...\n');
gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

% Get a sample of training features to compute normalization parameters
[sample_features, ~] = extract_sample_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size, 50);

% Compute normalization parameters from training data
feature_min = min(sample_features(:));
feature_max = max(sample_features(:));
feature_range = feature_max - feature_min;

fprintf('Normalization range: [%.4f, %.4f]\n', feature_min, feature_max);

%% Initialize video
videoObject = VideoReader('voilancevideo.mp4');
numberOfFrames = videoObject.NumberOfFrames;

%% Initialize cascade detector
option.detector = vision.CascadeObjectDetector('D:\Research Work\Fractional Order Legendre Polynomials\Libraries\gunDetectionUsingHog.xml');

%% Setup output video
outputVideo = VideoWriter('gun_detection_FLF_corrected.mp4');
outputVideo.FrameRate = videoObject.FrameRate;
open(outputVideo);

%% Initialize counters
totalMissedDetected = zeros(1, numberOfFrames);
totalGunDetected = zeros(1, numberOfFrames);
confidence_scores = [];

figure('Position', [100, 100, 800, 600]);

%% Process each frame
for frame = 1:min(numberOfFrames, 500)  % Process first 500 frames for testing
    img = read(videoObject, frame);
    
    fprintf('Processing frame %d/%d\n', frame, numberOfFrames);
    
    % Initial detection using cascade classifier
    bbox = step(option.detector, img);
    
    [rows, ~] = size(bbox);
    verified_bbox = [];
    
    if rows > 0
        missed = 0;
        gun = 0;
        
        for i = 1:rows
            % Extract and preprocess region
            thisRegion = imcrop(img, bbox(i,:));
            
            if size(thisRegion, 3) == 3
                thisRegion = rgb2gray(thisRegion);
            end
            
            thisRegion = imresize(thisRegion, image_size);
            thisRegion = double(thisRegion) / 255.0;
            
            % Extract FLF features
            features = [];
            for alpha = alpha_scales
                C = numeric_fractional_legendre_coef_matrix_2D(thisRegion, n, alpha, alpha);
                features = [features, C(:)'];
            end
            
            % CRITICAL: Normalize using training data parameters
            features_normalized = (features - feature_min) / feature_range;
            
            % Clip to [0,1] range
            features_normalized = max(0, min(1, features_normalized));
            
            % Predict
            [predictedLabel, score] = predict(svm_model, features_normalized);
            
            % Get confidence score if available
            if ~isempty(score)
                confidence = max(abs(score));
                confidence_scores = [confidence_scores, confidence];
            end
            
            if predictedLabel == 1
                verified_bbox = [verified_bbox; bbox(i,:)];
                gun = gun + 1;
                fprintf('  Gun detected! (Confidence: %.3f)\n', confidence);
            else
                missed = missed + 1;
                fprintf('  False positive discarded (Confidence: %.3f)\n', confidence);
            end
        end
        
        totalMissedDetected(frame) = missed;
        totalGunDetected(frame) = gun;
    end
    
    % Annotate frame
    if ~isempty(verified_bbox)
        detectedImg = insertShape(img, 'FilledRectangle', verified_bbox, 'Color', [230 30 14], 'Opacity', 0.3);
        detectedImg = insertText(detectedImg, verified_bbox(:,1:2), 'Gun (FLF)', 'AnchorPoint', 'LeftBottom', 'FontSize', 14);
    else
        detectedImg = img;
    end
    
    % Add frame info
    infoText = sprintf('Frame: %d | Guns: %d | Rejected: %d', frame, size(verified_bbox, 1), totalMissedDetected(frame));
    detectedImg = insertText(detectedImg, [10 10], infoText, 'FontSize', 14, 'BoxColor', 'white');
    
    % Write and display
    writeVideo(outputVideo, detectedImg);
    imagesc(detectedImg);
    title(sprintf('FLF Gun Detection - Frame %d/%d', frame, numberOfFrames));
    axis image off;
    drawnow;
end

close(outputVideo);

%% Display summary
fprintf('\n=== DETECTION SUMMARY ===\n');
fprintf('Frames processed: %d\n', frame);
fprintf('Total guns detected: %d\n', sum(totalGunDetected));
fprintf('Total false positives filtered: %d\n', sum(totalMissedDetected));
fprintf('Average confidence: %.3f\n', mean(confidence_scores));

disp('Video processing complete!');

%% Helper function to get sample features for normalization
function [features, labels] = extract_sample_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size, num_samples)
    % Extract a sample of features for normalization
    
    gun_files = dir(fullfile(gun_folder, '*.jpg'));
    gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];
    
    no_gun_files = dir(fullfile(no_gun_folder, '*.jpg'));
    no_gun_files = [no_gun_files; dir(fullfile(no_gun_folder, '*.png'))];
    
    % Sample subset
    num_gun_samples = min(num_samples/2, length(gun_files));
    num_nogun_samples = min(num_samples/2, length(no_gun_files));
    
    total_samples = num_gun_samples + num_nogun_samples;
    n_features = n * n * length(alpha_scales);
    features = zeros(total_samples, n_features);
    labels = zeros(total_samples, 1);
    
    idx = 1;
    
    % Sample gun images
    for i = 1:num_gun_samples
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
    
    % Sample no-gun images
    for i = 1:num_nogun_samples
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
    
    % Apply same normalization as training
    features = normalize(features, 'range');
end