function [features, labels, feature_dim, avg_keypoints] = extract_all_SURF_features(gun_folder, no_gun_folder, ...
    image_size, surf_threshold, num_features_per_image)
    % Extract SURF features using Bag of Features approach
    
    % Get image files - SORT for consistency
    gun_files = dir(fullfile(gun_folder, '*.jpg'));
    gun_files = [gun_files; dir(fullfile(gun_folder, '*.png'))];
    gun_files = sortfiles(gun_files);
    
    no_gun_files = dir(fullfile(no_gun_folder, '*.jpg'));
    no_gun_files = [no_gun_files; dir(fullfile(no_gun_folder, '*.png'))];
    no_gun_files = sortfiles(no_gun_files);
    
    total_images = length(gun_files) + length(no_gun_files);
    
    % First pass: collect all SURF descriptors for vocabulary building
    fprintf('Building visual vocabulary from SURF features...\n');
    all_descriptors = [];
    keypoint_counts = zeros(total_images, 1);
    idx = 1;
    
    % Process gun images
    fprintf('Extracting SURF from gun images: ');
    for i = 1:length(gun_files)
        img_path = fullfile(gun_folder, gun_files(i).name);
        [descriptors, num_kp] = extract_SURF_descriptors(img_path, image_size, surf_threshold);
        
        if ~isempty(descriptors)
            % Limit features per image
            if size(descriptors, 1) > num_features_per_image
                idx_sample = randperm(size(descriptors, 1), num_features_per_image);
                descriptors = descriptors(idx_sample, :);
            end
            all_descriptors = [all_descriptors; descriptors];
        end
        keypoint_counts(idx) = num_kp;
        idx = idx + 1;
        
        if mod(i, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' Done\n');
    
    % Process no-gun images
    fprintf('Extracting SURF from no-gun images: ');
    for i = 1:length(no_gun_files)
        img_path = fullfile(no_gun_folder, no_gun_files(i).name);
        [descriptors, num_kp] = extract_SURF_descriptors(img_path, image_size, surf_threshold);
        
        if ~isempty(descriptors)
            % Limit features per image
            if size(descriptors, 1) > num_features_per_image
                idx_sample = randperm(size(descriptors, 1), num_features_per_image);
                descriptors = descriptors(idx_sample, :);
            end
            all_descriptors = [all_descriptors; descriptors];
        end
        keypoint_counts(idx) = num_kp;
        idx = idx + 1;
        
        if mod(i, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' Done\n');
    
    % Build visual vocabulary using k-means
    vocab_size = 500;  % Size of visual vocabulary
    fprintf('Creating visual vocabulary with %d words...\n', vocab_size);
    [~, vocabulary] = kmeans(all_descriptors, vocab_size, ...
        'MaxIter', 200, 'Replicates', 1, 'Start', 'plus');
    
    % Second pass: create bag of features representation
    features = zeros(total_images, vocab_size);
    labels = zeros(total_images, 1);
    idx = 1;
    
    % Process gun images
    fprintf('Creating BoF representation for gun images: ');
    for i = 1:length(gun_files)
        img_path = fullfile(gun_folder, gun_files(i).name);
        features(idx, :) = create_bof_features(img_path, image_size, surf_threshold, vocabulary);
        labels(idx) = 1;  % Gun class
        idx = idx + 1;
        
        if mod(i, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' Done\n');
    
    % Process no-gun images
    fprintf('Creating BoF representation for no-gun images: ');
    for i = 1:length(no_gun_files)
        img_path = fullfile(no_gun_folder, no_gun_files(i).name);
        features(idx, :) = create_bof_features(img_path, image_size, surf_threshold, vocabulary);
        labels(idx) = 0;  % No-gun class
        idx = idx + 1;
        
        if mod(i, 10) == 0
            fprintf('.');
        end
    end
    fprintf(' Done\n');
    
    % Normalize features
    features = normalize(features, 'range');
    feature_dim = vocab_size;
    avg_keypoints = mean(keypoint_counts);
end
