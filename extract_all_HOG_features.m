function [features, labels, feature_dim] = extract_all_HOG_features(gun_folder, no_gun_folder, ...
    image_size, cell_size, block_size, num_bins, block_overlap)
    % Extract HOG features from all images
    
    % Get image files - SORT for consistency
    gun_files = dir(fullfile(gun_folder, '*.jpg'));
    gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];
    gun_files = sortfiles(gun_files);  % Sort by name
    
    no_gun_files = dir(fullfile(no_gun_folder, '*.jpg'));
    no_gun_files = [no_gun_files; dir(fullfile(no_gun_folder, '*.png'))];
    no_gun_files = sortfiles(no_gun_files);  % Sort by name
    
    % Get HOG feature dimension from a test image
    test_img = zeros(image_size);
    test_features = extractHOGFeatures(test_img, ...
        'CellSize', cell_size, ...
        'BlockSize', block_size, ...
        'NumBins', num_bins, ...
        'BlockOverlap', block_overlap);
    feature_dim = length(test_features);
    
    % Pre-allocate
    total_images = length(gun_files) + length(no_gun_files);
    features = zeros(total_images, feature_dim);
    labels = zeros(total_images, 1);
    
    idx = 1;
    
    % Process gun images
    fprintf('Processing gun images: ');
    for i = 1:length(gun_files)
        img_path = fullfile(gun_folder, gun_files(i).name);
        img = load_and_preprocess(img_path, image_size);
        features(idx, :) = extract_HOG_features(img, cell_size, block_size, num_bins, block_overlap);
        labels(idx) = 1;  % Gun class
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
        features(idx, :) = extract_HOG_features(img, cell_size, block_size, num_bins, block_overlap);
        labels(idx) = 0;  % No-gun class
        idx = idx + 1;
        
        if mod(i, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' Done\n');
    
    % Remove any empty rows
    features = features(1:idx-1, :);
    labels = labels(1:idx-1);
    
    % Normalize features with consistent method
    features = normalize(features, 'range');  % Normalize to [0,1] range
end

function sorted_files = sortfiles(files)
    % Sort files by name for consistency
    [~, idx] = sort({files.name});
    sorted_files = files(idx);
end
