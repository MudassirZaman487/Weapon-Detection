function analyze_feature_importance_FLF(X_train, y_train, n, alpha_scales)
    % Analyze which scale contributes most to classification
    
    n_features_per_scale = n * n;
    scale_accuracies = zeros(length(alpha_scales), 1);
    
    % Test each scale individually
    for i = 1:length(alpha_scales)
        % Extract features for this scale only
        start_idx = (i-1) * n_features_per_scale + 1;
        end_idx = i * n_features_per_scale;
        X_scale = X_train(:, start_idx:end_idx);
        
        % Train simple SVM
        model = fitcsvm(X_scale, y_train, 'KFold', 5);
        scale_accuracies(i) = 1 - kfoldLoss(model);
        
        fprintf('α = %.1f: Individual accuracy = %.2f%%\n', ...
            alpha_scales(i), scale_accuracies(i)*100);
    end
    
    % Plot scale importance
    figure('Position', [700, 100, 500, 400]);
    bar(alpha_scales, scale_accuracies*100);
    xlabel('Alpha (α)');
    ylabel('Classification Accuracy (%)');
    title('Classification Performance by Scale');
    grid on;
    ylim([0, 100]);
    
    % Add value labels on bars
    for i = 1:length(alpha_scales)
        text(alpha_scales(i), scale_accuracies(i)*100 + 1, ...
            sprintf('%.1f%%', scale_accuracies(i)*100), ...
            'HorizontalAlignment', 'center');
    end
end
