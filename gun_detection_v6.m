%% detect_guns_in_video.m
% Main script - automatically selects best method

clear; clc; close all;

%% Check video
videoFile = 'voilancevideo.mp4';
if ~exist(videoFile, 'file')
    error('Video file not found: %s', videoFile);
end

videoObject = VideoReader(videoFile);
totalFrames = videoObject.NumberOfFrames;
duration = totalFrames / videoObject.FrameRate;

fprintf('=== GUN DETECTION IN VIDEO ===\n');
fprintf('Video: %s\n', videoFile);
fprintf('Frames: %d\n', totalFrames);
fprintf('Duration: %.1f seconds (%.1f minutes)\n\n', duration, duration/60);

%% Ask user preference
fprintf('Detection Methods:\n');
fprintf('1. Complete (Most Accurate) - No cascade, multi-scale sliding window\n');
fprintf('2. Grid-based (Fast) - No cascade, grid search\n');
fprintf('3. Optimized (Balanced) - Motion detection + smart sampling\n');
fprintf('4. Fast with Cascade - Cascade + FLF verification\n');
fprintf('5. Auto-select based on video length\n\n');

choice = input('Select method (1-5): ');

%% Auto-select if chosen
if choice == 5
    if totalFrames < 1000
        choice = 1;  % Complete for short videos
        fprintf('Auto-selected: Complete method (short video)\n');
    elseif totalFrames < 5000
        choice = 2;  % Grid for medium videos
        fprintf('Auto-selected: Grid-based method (medium video)\n');
    else
        choice = 3;  % Optimized for long videos
        fprintf('Auto-selected: Optimized method (long video)\n');
    end
end

%% Check requirements
if ~exist('gun_detection_model_FLF.mat', 'file')
    error('FLF model not found! Run training first.');
end

if ~exist('FLF_normalization_params.mat', 'file')
    fprintf('Normalization parameters not found. Computing...\n');
    save_FLF_model_with_normalization;
end

%% Run selected method
fprintf('\nStarting detection...\n');

switch choice
    case 1
        fprintf('Running COMPLETE detection (most accurate)...\n');
        gun_detection_video_FLF_complete;
        
    case 2
        fprintf('Running GRID-BASED detection (fast)...\n');
        gun_detection_video_FLF_grid;
        
    case 3
        fprintf('Running OPTIMIZED detection (balanced)...\n');
        gun_detection_video_FLF_optimized;
        
    case 4
        fprintf('Running FAST detection with cascade...\n');
        if ~exist('Libraries/gunDetectionUsingHog.xml', 'file')
            error('Cascade classifier not found!');
        end
        gun_detection_video_FLF_fast_complete;
        
    otherwise
        error('Invalid choice!');
end

fprintf('\n✓ Detection complete! Check output video.\n');