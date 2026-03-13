function [accuracy, precision, recall, f1, cm] = evaluate_classifier(model, X_test, y_test)
    % Evaluate classifier performance
    
    % Make predictions
    y_pred = predict(model, X_test);
    
    % Calculate confusion matrix
    cm = confusionmat(y_test, y_pred);
    
    % Calculate metrics
    TP = cm(2,2);  % True Positives (correctly detected guns)
    TN = cm(1,1);  % True Negatives (correctly detected no-guns)
    FP = cm(1,2);  % False Positives (incorrectly detected as gun)
    FN = cm(2,1);  % False Negatives (missed guns)
    
    accuracy = (TP + TN) / sum(cm(:));
    
    if (TP + FP) > 0
        precision = TP / (TP + FP);
    else
        precision = 0;
    end
    
    if (TP + FN) > 0
        recall = TP / (TP + FN);
    else
        recall = 0;
    end
    
    if (precision + recall) > 0
        f1 = 2 * (precision * recall) / (precision + recall);
    else
        f1 = 0;
    end
end