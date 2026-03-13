function img = load_and_preprocess(img_path, target_size)
    % Load and preprocess a single image
    
    img = imread(img_path);
    
    % Convert to grayscale if needed
    if size(img, 3) == 3
        img = rgb2gray(img);
    end
    
    % Resize to standard size
    img = imresize(img, target_size);
    
    % Convert to double and normalize to [0, 1]
    img = double(img) / 255.0;
end
