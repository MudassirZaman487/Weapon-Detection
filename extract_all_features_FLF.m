function [features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size)
% Extract Fractional Legendre features for all images in the two folders.
% Depends on your own load_and_preprocess.m and extract_FLF_features.m

    % Collect files and sort by name for reproducibility
    gun_files = [dir(fullfile(gun_folder, '*.jpg')); dir(fullfile(gun_folder, '*.png'))];
    gun_files = sortfiles(gun_files);

    no_gun_files = [dir(fullfile(no_gun_folder, '*.jpg')); dir(fullfile(no_gun_folder, '*.png'))];
    no_gun_files = sortfiles(no_gun_files);

    total_images = length(gun_files) + length(no_gun_files);
    n_features   = n * n * length(alpha_scales);   % multi scale features

    features = zeros(total_images, n_features);
    labels   = zeros(total_images, 1);

    idx = 1;

    % Gun images
    fprintf('Processing gun images with FLF: ');
    for i = 1:length(gun_files)
        img_path = fullfile(gun_folder, gun_files(i).name);
        img = load_and_preprocess(img_path, image_size);          % user provided
        features(idx, :) = extract_FLF_features(img, n, alpha_scales);  % user provided
        labels(idx) = 1;
        idx = idx + 1;
        if mod(i,10)==0, fprintf('.'); end
    end
    fprintf(' Done\n');

    % No gun images
    fprintf('Processing no-gun images with FLF: ');
    for i = 1:length(no_gun_files)
        img_path = fullfile(no_gun_folder, no_gun_files(i).name);
        img = load_and_preprocess(img_path, image_size);          % user provided
        features(idx, :) = extract_FLF_features(img, n, alpha_scales);  % user provided
        labels(idx) = 0;
        idx = idx + 1;
        if mod(i,10)==0, fprintf('.'); end
    end
    fprintf(' Done\n');

    % Trim any unused rows
    features = features(1:idx-1, :);
    labels   = labels(1:idx-1);

    % IMPORTANT: do not normalize here. We will scale after the split to avoid leakage.
end


