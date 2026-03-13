%% gun_detection_video_FLF_fast_complete.m
% Fast version using cascade classifier + FLF verification
% Processes entire video quickly

clear; clc; close all;

%% Load FLF model
load('gun_detection_model_FLF.mat');

%% Load normalization parameters
if exist('FLF_normalization_params.mat', 'file')
    load('FLF_normalization_params.mat');
else
    % Compute normalization parameters
    fprintf('Computing normalization parameters...\n');
    gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
    no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';
    
    [all_features, ~] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);
    
    feature_mins = min(all_features, [], 1);
    feature_maxs = max(all_features, [], 1);
    feature_ranges = feature_maxs - feature_mins;
    feature_ranges(feature_ranges == 0) = 1;
    
    save('FLF_normalization_params.mat', 'feature_mins', 'feature_maxs', 'feature_ranges');
    clear all_features;
end

%% Initialize video and cascade detector
videoObject = VideoReader('voilancevideo.mp4');
totalFrames = videoObject.NumberOfFrames;
detector = vision.CascadeObjectDetector('Libraries/gunDetectionUsingHog.xml');

%% Output video
outputVideo = VideoWriter('gun_detection_FLF_fast_complete.mp4', 'MPEG-4');
outputVideo.FrameRate = videoObject.FrameRate;
outputVideo.Quality = 95;
open(outputVideo);

%% Initialize counters
guns_detected = 0;
false_positives = 0;
detection_log = [];

%% Create display figure
figure('Position', [100, 100, 1000, 600]);

%% Process all frames
fprintf('\nProcessing %d frames with cascade + FLF verification...\n', totalFrames);
startTime = tic;

for frameNum = 1:totalFrames
    % Read frame
    frame = read(videoObject, frameNum);
    
    % Cascade detection
    bbox = step(detector, frame);
    
    verified_bbox = [];
    frame_guns = 0;
    frame_fp = 0;
    
    if ~isempty(bbox)
        for i = 1:size(bbox, 1)
            % Extract region
            region = imcrop(frame, bbox(i,:));
            
            % Preprocess
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
            
            % Normalize
            features_normalized = (features - feature_mins) ./ feature_ranges;
            features_normalized = max(0, min(1, features_normalized));
            
            % Predict
            [prediction, score] = predict(svm_model, features_normalized);
            
            if prediction == 1
                verified_bbox = [verified_bbox; bbox(i,:)];
                frame_guns = frame_guns + 1;
                guns_detected = guns_detected + 1;
            else
                frame_fp = frame_fp + 1;
                false_positives = false_positives + 1;
            end
        end
    end
    
    % Store log
    detection_log(frameNum).frame = frameNum;
    detection_log(frameNum).initial = size(bbox, 1);
    detection_log(frameNum).verified = frame_guns;
    detection_log(frameNum).rejected = frame_fp;
    
    % Annotate frame
    annotated = frame;
    
    % Show verified detections
    for i = 1:size(verified_bbox, 1)
        annotated = insertShape(annotated, 'Rectangle', verified_bbox(i,:), ...
            'Color', 'red', 'LineWidth', 3);
        annotated = insertText(annotated, verified_bbox(i,1:2), 'GUN', ...
            'FontSize', 16, 'BoxColor', 'red', 'TextColor', 'white');
    end
    
    % Add stats
    stats_text = sprintf('Frame %d/%d | Guns: %d | FP filtered: %d', ...
        frameNum, totalFrames, frame_guns, frame_fp);
    annotated = insertText(annotated, [10 10], stats_text, ...
        'FontSize', 14, 'BoxColor', 'black', 'TextColor', 'white');
    
    % Display periodically
    if mod(frameNum, 30) == 1 || frame_guns > 0
        imshow(annotated);
        title(sprintf('Gun Detection - Frame %d/%d', frameNum, totalFrames));
        drawnow;
    end
    
    % Write frame
    writeVideo(outputVideo, annotated);
    
    % Progress
    if mod(frameNum, 100) == 0
        elapsed = toc(startTime);
        fps = frameNum / elapsed;
        eta = (totalFrames - frameNum) / fps;
        fprintf('Progress: %d/%d (%.1f%%) | FPS: %.1f | ETA: %.1f min\n', ...
            frameNum, totalFrames, frameNum/totalFrames*100, fps, eta/60);
    end
end

close(outputVideo);
totalTime = toc(startTime);

%% Summary
fprintf('\n========== FINAL SUMMARY ==========\n');
fprintf('Total frames: %d\n', totalFrames);
fprintf('Processing time: %.2f minutes\n', totalTime/60);
fprintf('Average FPS: %.2f\n', totalFrames/totalTime);
fprintf('Total guns detected: %d\n', guns_detected);
fprintf('False positives filtered: %d\n', false_positives);
fprintf('Detection rate: %.2f%%\n', guns_detected/(guns_detected+false_positives)*100);

frames_with_guns = sum([detection_log.verified] > 0);
fprintf('Frames with guns: %d (%.2f%%)\n', frames_with_guns, frames_with_guns/totalFrames*100);

%% Save results
save('FLF_video_detection_results.mat', 'detection_log', 'guns_detected', ...
    'false_positives', 'totalTime');

fprintf('\nResults saved. Output video: gun_detection_FLF_fast_complete.mp4\n');

%% Helper function
function [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size)
    gun_files = [dir(fullfile(gun_folder, '*.jpg')); dir(fullfile(gun_folder, '*.png'))];
    no_gun_files = [dir(fullfile(no_gun_folder, '*.jpg')); dir(fullfile(no_gun_folder, '*.png'))];
    
    total_images = length(gun_files) + length(no_gun_files);
    n_features = n * n * length(alpha_scales);
    features = zeros(total_images, n_features);
    labels = zeros(total_images, 1);
    
    idx = 1;
    % Process files...
    for i = 1:length(gun_files)
        img = imread(fullfile(gun_folder, gun_files(i).name));
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
    
    for i = 1:length(no_gun_files)
        img = imread(fullfile(no_gun_folder, no_gun_files(i).name));
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