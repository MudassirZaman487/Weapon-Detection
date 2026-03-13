%% gun_detection_video_FLF_complete.m
% Complete gun detection using ONLY FLF features (no cascade classifier)
% More accurate but slower - uses sliding window approach

clear; clc; close all;

%% Load FLF model and normalization parameters
fprintf('Loading FLF model and normalization parameters...\n');
load('gun_detection_model_FLF.mat');

% Load or compute normalization parameters
if exist('FLF_normalization_params.mat', 'file')
    load('FLF_normalization_params.mat');
    fprintf('Normalization parameters loaded from file.\n');
else
    fprintf('Computing normalization parameters...\n');
    gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
    no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';
    
    [all_features, ~] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);
    
    feature_mins = min(all_features, [], 10);
    feature_maxs = max(all_features, [], 10);
    feature_ranges = feature_maxs - feature_mins;
    feature_ranges(feature_ranges == 0) = 1;
    
    save('FLF_normalization_params.mat', 'feature_mins', 'feature_maxs', 'feature_ranges');
    clear all_features;
end

%% Video parameters
videoFile = 'voilancevideo.mp4';
videoObject = VideoReader(videoFile);
totalFrames = videoObject.NumberOfFrames;

%% Sliding window parameters
window_sizes = [64 64; 96 96; 128 128];  % Multi-scale detection
stride_ratio = 0.25;  % 25% of window size
detection_threshold = 0.5;  % Confidence threshold

%% Output video setup
outputVideo = VideoWriter('gun_detection_FLF_complete.mp4', 'MPEG-4');
outputVideo.FrameRate = videoObject.FrameRate;
outputVideo.Quality = 95;
open(outputVideo);

%% Initialize results storage
detection_log = struct('frame', {}, 'detections', {}, 'time', {});
total_detections = 0;

%% Create figure for display
fig = figure('Position', [100, 100, 1200, 600]);

%% Process all frames
fprintf('\nProcessing %d frames...\n', totalFrames);
fprintf('This may take a while for full video processing.\n\n');

startTime = tic;

for frameNum = 1:totalFrames
    frameStartTime = tic;
    
    % Read frame
    frame = read(videoObject, frameNum);
    grayFrame = rgb2gray(frame);
    
    % Multi-scale sliding window detection
    all_detections = [];
    
    for scale_idx = 1:size(window_sizes, 1)
        window_size = window_sizes(scale_idx, :);
        stride = round(window_size(1) * stride_ratio);
        
        % Skip if window is larger than frame
        if window_size(1) > size(grayFrame, 1) || window_size(2) > size(grayFrame, 2)
            continue;
        end
        
        % Sliding window
        for y = 1:stride:(size(grayFrame, 1) - window_size(1) + 1)
            for x = 1:stride:(size(grayFrame, 2) - window_size(2) + 1)
                % Extract window
                window = grayFrame(y:y+window_size(1)-1, x:x+window_size(2)-1);
                
                % Resize to model input size
                window_resized = imresize(window, image_size);
                window_norm = double(window_resized) / 255.0;
                
                % Extract FLF features
                features = [];
                for alpha = alpha_scales
                    C = numeric_fractional_legendre_coef_matrix_2D(window_norm, n, alpha, alpha);
                    features = [features, C(:)'];
                end
                
                % Apply normalization
                features_normalized = (features - feature_mins) ./ feature_ranges;
                features_normalized = max(0, min(1, features_normalized));
                
                % Predict
                [label, score] = predict(svm_model, features_normalized);
                
                if label == 1 && abs(score(2)) > detection_threshold
                    all_detections = [all_detections; x, y, window_size(2), window_size(1), abs(score(2))];
                end
            end
        end
    end
    
    % Non-maximum suppression
    if ~isempty(all_detections)
        final_detections = nms_detections(all_detections, 0.3);
        num_detections = size(final_detections, 1);
    else
        final_detections = [];
        num_detections = 0;
    end
    
    % Store results
    detection_log(frameNum).frame = frameNum;
    detection_log(frameNum).detections = final_detections;
    detection_log(frameNum).time = toc(frameStartTime);
    
    total_detections = total_detections + num_detections;
    
    % Annotate frame
    annotatedFrame = frame;
    for i = 1:size(final_detections, 1)
        bbox = final_detections(i, 1:4);
        confidence = final_detections(i, 5);
        
        % Draw rectangle
        annotatedFrame = insertShape(annotatedFrame, 'Rectangle', bbox, ...
            'Color', 'red', 'LineWidth', 3);
        
        % Add text
        label_text = sprintf('GUN %.2f', confidence);
        annotatedFrame = insertText(annotatedFrame, [bbox(1), bbox(2)-15], ...
            label_text, 'FontSize', 14, 'BoxColor', 'red', 'TextColor', 'white');
    end
    
    % Add frame info
    info_text = sprintf('Frame %d/%d | Guns: %d | Time: %.2fs', ...
        frameNum, totalFrames, num_detections, detection_log(frameNum).time);
    annotatedFrame = insertText(annotatedFrame, [10 10], info_text, ...
        'FontSize', 16, 'BoxColor', 'black', 'TextColor', 'white');
    
    % Display (update every 10 frames to save time)
    if mod(frameNum, 10) == 1 || num_detections > 0
        imshow(annotatedFrame);
        title(sprintf('FLF Gun Detection - Frame %d/%d', frameNum, totalFrames));
        drawnow;
    end
    
    % Write to output video
    writeVideo(outputVideo, annotatedFrame);
    
    % Progress update
    if mod(frameNum, 100) == 0
        elapsed = toc(startTime);
        fps = frameNum / elapsed;
        eta = (totalFrames - frameNum) / fps;
        fprintf('Progress: %d/%d frames (%.1f%%) | FPS: %.1f | ETA: %.1f min\n', ...
            frameNum, totalFrames, frameNum/totalFrames*100, fps, eta/60);
    end
end

close(outputVideo);
totalTime = toc(startTime);

%% Generate summary report
fprintf('\n\n========== DETECTION SUMMARY ==========\n');
fprintf('Total frames processed: %d\n', totalFrames);
fprintf('Total processing time: %.2f seconds (%.2f minutes)\n', totalTime, totalTime/60);
fprintf('Average FPS: %.2f\n', totalFrames/totalTime);
fprintf('Total guns detected: %d\n', total_detections);

% Find frames with detections
frames_with_guns = find(arrayfun(@(x) ~isempty(x.detections), detection_log));
fprintf('Frames with guns: %d (%.2f%%)\n', length(frames_with_guns), ...
    length(frames_with_guns)/totalFrames*100);

%% Plot detection timeline
figure('Position', [100, 100, 1200, 400]);

% Detection count per frame
detection_counts = arrayfun(@(x) size(x.detections, 1), detection_log);

subplot(2,1,1);
plot(1:totalFrames, detection_counts, 'b-', 'LineWidth', 1);
xlabel('Frame Number');
ylabel('Detections');
title('Gun Detections Timeline');
grid on;
xlim([1 totalFrames]);

% Processing time per frame
frame_times = arrayfun(@(x) x.time, detection_log);

subplot(2,1,2);
plot(1:totalFrames, frame_times, 'r-', 'LineWidth', 1);
xlabel('Frame Number');
ylabel('Processing Time (s)');
title('Processing Time per Frame');
grid on;
xlim([1 totalFrames]);

%% Save detection log
save('FLF_detection_log.mat', 'detection_log', 'total_detections', 'totalTime');
fprintf('\nDetection log saved to FLF_detection_log.mat\n');
fprintf('Output video saved to gun_detection_FLF_complete.mp4\n');

%% Helper Functions

function merged = nms_detections(detections, overlap_threshold)
    % Non-maximum suppression for overlapping detections
    if isempty(detections)
        merged = [];
        return;
    end
    
    % Sort by confidence
    [~, idx] = sort(detections(:,5), 'descend');
    detections = detections(idx, :);
    
    merged = [];
    while ~isempty(detections)
        % Take highest confidence detection
        merged = [merged; detections(1,:)];
        
        % Remove overlapping detections
        overlaps = calculate_overlaps(detections(1,1:4), detections(2:end,1:4));
        keep = overlaps < overlap_threshold;
        detections = detections([false; keep], :);
    end
end

function overlaps = calculate_overlaps(box1, boxes)
    % Calculate IoU between one box and multiple boxes
    overlaps = zeros(size(boxes, 1), 1);
    
    for i = 1:size(boxes, 1)
        box2 = boxes(i, :);
        
        % Intersection
        x1 = max(box1(1), box2(1));
        y1 = max(box1(2), box2(2));
        x2 = min(box1(1)+box1(3), box2(1)+box2(3));
        y2 = min(box1(2)+box1(4), box2(2)+box2(4));
        
        if x2 > x1 && y2 > y1
            intersection = (x2-x1) * (y2-y1);
            area1 = box1(3) * box1(4);
            area2 = box2(3) * box2(4);
            union = area1 + area2 - intersection;
            overlaps(i) = intersection / union;
        end
    end
end

function [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size)
    % Extract features for normalization
    gun_files = [dir(fullfile(gun_folder, '*.jpg')); dir(fullfile(gun_folder, '*.png'))];
    no_gun_files = [dir(fullfile(no_gun_folder, '*.jpg')); dir(fullfile(no_gun_folder, '*.png'))];
    
    total_images = length(gun_files) + length(no_gun_files);
    n_features = n * n * length(alpha_scales);
    features = zeros(total_images, n_features);
    labels = zeros(total_images, 1);
    
    idx = 1;
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