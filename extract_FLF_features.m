
function features = extract_FLF_features(img, n, alpha_scales)
    % Extract multi-scale Fractional Legendre features from a single image
    
    features = [];
    
    % Extract features at each scale
    for alpha = alpha_scales
        % Compute Fractional Legendre coefficients
        C = numeric_fractional_legendre_coef_matrix_2D(img, n, alpha, alpha);
        
        % Flatten and concatenate
        features = [features, C(:)'];
    end
end
