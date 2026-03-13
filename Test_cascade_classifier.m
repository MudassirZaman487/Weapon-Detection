%% test_cascade_detector.m
% Test if cascade detector is finding actual guns

clear; clc; close all;

%% Test cascade detector on known gun images
detector = vision.CascadeObjectDetector('Libraries/gunDetectionUsingHog.xml');

gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
gun_files = dir(fullfile(gun_folder, '*.jpg'));
gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];

fprintf('Testing cascade detector on known gun images...\n\n');

detected_count = 0;
for i = 1:min(20, length(gun_files))
    img = imread(fullfile(gun_folder, gun_files(i).name));
    bbox = step(detector, img);
    
    if ~isempty(bbox)
        detected_count = detected_count + 1;
        fprintf('Image %d: DETECTED %d gun(s)\n', i, size(bbox, 1));
        
        % Show the detection
        figure;
        imshow(img);
        hold on;
        for j = 1:size(bbox, 1)
            rectangle('Position', bbox(j,:), 'EdgeColor', 'r', 'LineWidth', 2);
        end
        title(sprintf('Gun Image %d - %d detections', i, size(bbox, 1)));
        pause(0.5);
    else
        fprintf('Image %d: NOT DETECTED\n', i);
    end
end

fprintf('\n\nCascade detector found guns in %d/%d images (%.1f%%)\n', ...
    detected_count, min(20, length(gun_files)), ...
    detected_count/min(20, length(gun_files))*100);

%% If cascade detector is not finding guns, we need a different approach
if detected_count < 5
    fprintf('\n*** CASCADE DETECTOR IS NOT WORKING WELL ***\n');
    fprintf('Consider using sliding window approach instead.\n');
end