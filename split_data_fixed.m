
function [X_train, X_test, y_train, y_test, train_idx, test_idx] = split_data_fixed(features, labels, train_ratio)
    % Split data with FIXED random seed
    
    % Set seed
    rng(21);
    
    % Random permutation
    n_samples = size(features, 1);
    idx = randperm(n_samples);
    
    % Split indices
    n_train = round(n_samples * train_ratio);
    train_idx = idx(1:n_train);
    test_idx = idx(n_train+1:end);
    
    % Sort indices for consistency
    train_idx = sort(train_idx);
    test_idx = sort(test_idx);
    
    % Split data
    X_train = features(train_idx, :);
    X_test = features(test_idx, :);
    y_train = labels(train_idx);
    y_test = labels(test_idx);
end

