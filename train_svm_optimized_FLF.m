
function svm_model = train_svm_optimized_FLF(X_train, y_train)
    % Train SVM with optimization for FLF features
    
    % Set random seed again for SVM internal randomness
    rng(21);
    
    % Define hyperparameter ranges for optimization
    box_values = [0.1, 1, 10, 100];
    kernel_scale_values = {0.1, 1, 10, 'auto'};  % Cell array to handle 'auto'
    
    best_accuracy = 0;
    best_box = 10;
    best_ks = 'auto';
    
    fprintf('Optimizing SVM hyperparameters...\n');
    
    % Simple grid search with cross-validation
    for box = box_values
        for ks_idx = 1:length(kernel_scale_values)
            ks = kernel_scale_values{ks_idx};
            
            % Train SVM with current parameters
            try
                temp_model = fitcsvm(X_train, y_train, ...
                    'KernelFunction', 'rbf', ...
                    'Standardize', true, ...
                    'BoxConstraint', box, ...
                    'KernelScale', ks, ...
                    'Solver', 'SMO', ...
                    'KFold', 5);  % 5-fold cross-validation
                
                % Get cross-validation accuracy
                cv_accuracy = 1 - kfoldLoss(temp_model);
                
                if cv_accuracy > best_accuracy
                    best_accuracy = cv_accuracy;
                    best_box = box;
                    best_ks = ks;
                    fprintf('  New best: BoxConstraint=%.1f, KernelScale=%s, CV Accuracy=%.4f\n', ...
                        box, num2str(ks), cv_accuracy);
                end
            catch
                % Skip invalid parameter combinations
                continue;
            end
        end
    end
    
    % Train final model with best parameters
    fprintf('Training final model with best parameters...\n');
    svm_model = fitcsvm(X_train, y_train, ...
        'KernelFunction', 'rbf', ...
        'Standardize', true, ...
        'BoxConstraint', best_box, ...
        'KernelScale', best_ks, ...
        'Solver', 'SMO', ...
        'CacheSize', 'maximal');
    
    fprintf('Best parameters: BoxConstraint=%.1f, KernelScale=%s\n', best_box, num2str(best_ks));
end
