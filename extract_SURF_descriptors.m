function [descriptors, num_keypoints] = extract_SURF_descriptors(img_path, target_size, threshold)
    % Extract SURF descriptors from a single image
    
    img = imread(img_path);
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Resize to standard size
    img = imresize(img, target_size);
    
    % Detect SURF features
    points = detectSURFFeatures(img, 'MetricThreshold', threshold);
    
    % Extract descriptors
    [descriptors, ~] = extractFeatures(img, points, 'Method', 'SURF');
    
    num_keypoints = length(points);
end