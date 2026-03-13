%% compare_all_classifiers_FLF.m
% Compare several light classifiers on the same FLF features
% SVM RBF, Random Forest, kNN, Neural Net, Decision Tree, Naive Bayes
% MATLAB R2014b friendly

clc; clear; close all;
rng(21);   % fixed seed

%% Parameters
n            = 10;                % Legendre count per scale
alpha_scales = [0.5 1.0 1.5];     % fractional orders
image_size   = [64 64];
train_ratio  = 0.8;

fprintf('COMPARATIVE ANALYSIS OF CLASSIFIERS FOR GUN DETECTION WITH FLF\n');
fprintf('===============================================================\n');
fprintf('Polynomial order: %d x %d = %d features per scale\n', n, n, n*n);
fprintf('Alpha scales: [%s]\n', num2str(alpha_scales));
fprintf('Total features: %d\n\n', n*n*length(alpha_scales));

%% Data paths  edit to your folders
gun_folder    = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Positive';
no_gun_folder = 'D:\Research Work\Adnan bhai Research Work\detection\New dataset\Negative';

%% Extract features once
fprintf('Loading images and extracting FLF features...\n');
tic;
[features, labels] = extract_all_features_FLF(gun_folder, no_gun_folder, n, alpha_scales, image_size);
feat_time = toc;

fprintf('Total samples: %d\n', length(labels));
fprintf('Gun samples: %d\n', sum(labels==1));
fprintf('No-gun samples: %d\n', sum(labels==0));
fprintf('Feature extraction time: %.2f seconds\n\n', feat_time);

%% Split set fixed and scale using train stats only
[X_train, X_test, y_train, y_test, train_idx, test_idx] = split_data_fixed(features, labels, train_ratio);

[X_train, fmin, fmax] = minmax_scale_2014b(X_train);   % fit on train
X_test = apply_minmax_2014b(X_test, fmin, fmax);       % apply to test

fprintf('Training samples: %d\n', size(X_train,1));
fprintf('Testing samples: %d\n\n', size(X_test,1));

classifier_names = {'SVM','Random Forest','k-NN','Neural Network','Decision Tree','Naive Bayes'};
results = struct();

%% 1 SVM RBF
fprintf('1. Training SVM RBF ...\n');
mem_before = safe_memory_struct();
tic;
svm_model = train_svm_optimized_FLF(X_train, y_train);     % defined below
results.SVM.training_time = toc;
mem_after  = safe_memory_struct();
results.SVM.memory_usage_MB = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB)/1e6;

tic;
svm_pred = predict(svm_model, X_test);
test_time = toc;
results.SVM.time_per_image = test_time / size(X_test,1);
[acc, prec, rec, f1, cm] = metrics_from_labels(y_test, svm_pred);
results.SVM.accuracy=acc; results.SVM.precision=prec; results.SVM.recall=rec; results.SVM.f1=f1; results.SVM.cm=cm;
fprintf('   Done! Accuracy: %.2f%%\n\n', 100*acc);

%% 2 Random Forest
fprintf('2. Training Random Forest 100 trees ...\n');
mem_before = safe_memory_struct();
tic;
rf_model = TreeBagger(100, X_train, y_train, 'Method','classification', ...
                      'NumPredictorsToSample', floor(sqrt(size(X_train,2))));
results.RF.training_time = toc;
mem_after  = safe_memory_struct();
results.RF.memory_usage_MB = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB)/1e6;

tic;
rf_pred = predict(rf_model, X_test);
rf_pred = str2double(rf_pred);
test_time = toc;
results.RF.time_per_image = test_time / size(X_test,1);
[acc, prec, rec, f1, cm] = metrics_from_labels(y_test, rf_pred);
results.RF.accuracy=acc; results.RF.precision=prec; results.RF.recall=rec; results.RF.f1=f1; results.RF.cm=cm;
fprintf('   Done! Accuracy: %.2f%%\n\n', 100*acc);

%% 3 kNN
fprintf('3. Training kNN k=5 ...\n');
mem_before = safe_memory_struct();
tic;
knn_model = fitcknn(X_train, y_train, 'NumNeighbors',5, 'Distance','euclidean', 'DistanceWeight','inverse');
results.KNN.training_time = toc;
mem_after  = safe_memory_struct();
results.KNN.memory_usage_MB = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB)/1e6;

tic;
knn_pred = predict(knn_model, X_test);
test_time = toc;
results.KNN.time_per_image = test_time / size(X_test,1);
[acc, prec, rec, f1, cm] = metrics_from_labels(y_test, knn_pred);
results.KNN.accuracy=acc; results.KNN.precision=prec; results.KNN.recall=rec; results.KNN.f1=f1; results.KNN.cm=cm;
fprintf('   Done! Accuracy: %.2f%%\n\n', 100*acc);

%% 4 Neural Net
fprintf('4. Training Neural Network 150-75 ...\n');
mem_before = safe_memory_struct();
tic;
hiddenLayerSize = [150 75];
net = patternnet(hiddenLayerSize);
net.trainParam.showWindow = false;

% use our external split only
net.divideFcn = 'divideind';
Ntr = size(X_train,1);
net.divideParam.trainInd = 1:Ntr;
net.divideParam.valInd   = [];
net.divideParam.testInd  = [];

y_train_nn = full(ind2vec(y_train' + 1));   % classes 1 and 2
net = train(net, X_train', y_train_nn);
results.NN.training_time = toc;
mem_after  = safe_memory_struct();
results.NN.memory_usage_MB = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB)/1e6;

tic;
y_pred_nn = net(X_test');
[~, pred_idx] = max(y_pred_nn);
nn_pred = pred_idx' - 1;
test_time = toc;
results.NN.time_per_image = test_time / size(X_test,1);
[acc, prec, rec, f1, cm] = metrics_from_labels(y_test, nn_pred);
results.NN.accuracy=acc; results.NN.precision=prec; results.NN.recall=rec; results.NN.f1=f1; results.NN.cm=cm;
fprintf('   Done! Accuracy: %.2f%%\n\n', 100*acc);

%% 5 Decision Tree
fprintf('5. Training Decision Tree ...\n');
mem_before = safe_memory_struct();
tic;
dt_model = fitctree(X_train, y_train, 'SplitCriterion','gdi', 'MinLeafSize',5, 'MaxNumSplits',100);
results.DT.training_time = toc;
mem_after  = safe_memory_struct();
results.DT.memory_usage_MB = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB)/1e6;

tic;
dt_pred = predict(dt_model, X_test);
test_time = toc;
results.DT.time_per_image = test_time / size(X_test,1);
[acc, prec, rec, f1, cm] = metrics_from_labels(y_test, dt_pred);
results.DT.accuracy=acc; results.DT.precision=prec; results.DT.recall=rec; results.DT.f1=f1; results.DT.cm=cm;
fprintf('   Done! Accuracy: %.2f%%\n\n', 100*acc);

%% 6 Naive Bayes
fprintf('6. Training Naive Bayes ...\n');
mem_before = safe_memory_struct();
tic;
nb_model = fitcnb(X_train, y_train, 'DistributionNames','normal', 'Prior','empirical');
results.NB.training_time = toc;
mem_after  = safe_memory_struct();
results.NB.memory_usage_MB = (mem_after.MemUsedMATLAB - mem_before.MemUsedMATLAB)/1e6;

tic;
nb_pred = predict(nb_model, X_test);
test_time = toc;
results.NB.time_per_image = test_time / size(X_test,1);
[acc, prec, rec, f1, cm] = metrics_from_labels(y_test, nb_pred);
results.NB.accuracy=acc; results.NB.precision=prec; results.NB.recall=rec; results.NB.f1=f1; results.NB.cm=cm;
fprintf('   Done! Accuracy: %.2f%%\n\n', 100*acc);

%% Build comparison table
fprintf('\n============================================================\n');
fprintf('                 CLASSIFIER COMPARISON TABLE                 \n');
fprintf('============================================================\n\n');

field_names = {'SVM','RF','KNN','NN','DT','NB'};
table_data = zeros(numel(field_names), 7);

for i = 1:length(field_names)
    R = results.(field_names{i});
    table_data(i,1) = R.accuracy * 100;
    table_data(i,2) = R.precision * 100;
    table_data(i,3) = R.recall * 100;
    table_data(i,4) = R.f1 * 100;
    table_data(i,5) = R.training_time;
    table_data(i,6) = R.time_per_image * 1000;  % ms
    table_data(i,7) = R.memory_usage_MB;
end

T = table(classifier_names', table_data(:,1), table_data(:,2), table_data(:,3), ...
          table_data(:,4), table_data(:,5), table_data(:,6), table_data(:,7), ...
          'VariableNames', {'Classifier','Accuracy_pct','Precision_pct','Recall_pct', ...
                            'F1_Score_pct','Training_Time_s','Time_per_Image_ms','Memory_Usage_MB'});
disp(T);

[best_acc, best_acc_idx]         = max(table_data(:,1));
[best_f1,  best_f1_idx]          = max(table_data(:,4));
[fastest_train, fastest_train_idx]= min(table_data(:,5));
[fastest_test,  fastest_test_idx] = min(table_data(:,6));
[min_memory,   min_memory_idx]    = min(table_data(:,7));

fprintf('\n============================================================\n');
fprintf('                      BEST PERFORMERS                        \n');
fprintf('============================================================\n');
fprintf('Highest Accuracy:        %s (%.2f%%)\n', classifier_names{best_acc_idx}, best_acc);
fprintf('Highest F1-Score:        %s (%.2f%%)\n', classifier_names{best_f1_idx}, best_f1);
fprintf('Fastest Training:        %s (%.2f s)\n', classifier_names{fastest_train_idx}, fastest_train);
fprintf('Fastest Prediction:      %s (%.4f ms/image)\n', classifier_names{fastest_test_idx}, fastest_test);
fprintf('Lowest Memory Usage:     %s (%.2f MB)\n', classifier_names{min_memory_idx}, min_memory);

%% Plots  no sgtitle in 2014b, use annotation for supertitle
figure('Position',[100 100 1200 800]);

subplot(2,3,1);
bar(table_data(:,1));
set(gca,'XTick',1:numel(classifier_names),'XTickLabel',classifier_names);
ylabel('Accuracy (%)');
title('Accuracy');
grid on;

subplot(2,3,2);
bar(table_data(:,4));
set(gca,'XTick',1:numel(classifier_names),'XTickLabel',classifier_names);
ylabel('F1-Score (%)');
title('F1-Score');
grid on;

subplot(2,3,3);
bar(table_data(:,5));
set(gca,'XTick',1:numel(classifier_names),'XTickLabel',classifier_names);
ylabel('Training Time (s)');
title('Training Time');
grid on;

subplot(2,3,4);
bar(table_data(:,6));
set(gca,'XTick',1:numel(classifier_names),'XTickLabel',classifier_names);
ylabel('Time per Image (ms)');
title('Prediction Speed');
grid on;

subplot(2,3,5);
bar(table_data(:,7));
set(gca,'XTick',1:numel(classifier_names),'XTickLabel',classifier_names);
ylabel('Memory (MB)');
title('Memory Usage');
grid on;

subplot(2,3,6);
plot(table_data(:,3), table_data(:,2), 'o-', 'LineWidth',2, 'MarkerSize',8);
text(table_data(:,3)+0.5, table_data(:,2), classifier_names, 'FontSize',10);
xlabel('Recall (%)'); ylabel('Precision (%)');
title('Precision vs Recall');
grid on;

annotation('textbox',[0 0.95 1 0.04],'String','Gun Detection with FLF Features  Classifier Comparison', ...
           'EdgeColor','none','HorizontalAlignment','center','FontWeight','bold');

%% Save results
save('classifier_comparison_FLF.mat','results','T','classifier_names','table_data');
fprintf('\nSaved to classifier_comparison_FLF.mat\n');

% =======================================================================
%                               FUNCTIONS
% =======================================================================

function [Xtr, Xte, ytr, yte, tr_idx, te_idx] = split_data_fixed(X, y, train_ratio)
% Stratified split with fixed seed
    n = length(y);
    pos = find(y==1); neg = find(y==0);
    ntr_pos = floor(train_ratio * numel(pos));
    ntr_neg = floor(train_ratio * numel(neg));

    pos = pos(randperm(numel(pos)));
    neg = neg(randperm(numel(neg)));

    tr_idx = [pos(1:ntr_pos); neg(1:ntr_neg)];
    te_idx = [pos(ntr_pos+1:end); neg(ntr_neg+1:end)];

    % Keep a fixed order
    tr_idx = tr_idx(randperm(numel(tr_idx)));
    te_idx = te_idx(randperm(numel(te_idx)));

    Xtr = X(tr_idx, :);
    ytr = y(tr_idx);
    Xte = X(te_idx, :);
    yte = y(te_idx);
end

function [Xn, fmin, fmax] = minmax_scale_2014b(X)
% Column wise [0,1] scaling
    fmin = min(X, [], 1);
    fmax = max(X, [], 1);
    denom = fmax - fmin;
    denom(denom==0) = 1;
    Xn = bsxfun(@rdivide, bsxfun(@minus, X, fmin), denom);
end

function Xn = apply_minmax_2014b(X, fmin, fmax)
    denom = fmax - fmin;
    denom(denom==0) = 1;
    Xn = bsxfun(@rdivide, bsxfun(@minus, X, fmin), denom);
end

function mem = safe_memory_struct()
% memory() exists on Windows. Wrap for safety.
    try
        mem = memory;
    catch
        mem.MemUsedMATLAB = NaN;
    end
end

function [acc, prec, rec, f1, cm] = metrics_from_labels(y_true, y_pred)
% Return accuracy, precision, recall, F1, confusion matrix for binary 0/1
    cm = confusionmat(y_true, y_pred);
    % cm rows true [0;1], cols pred [0 1]
    tn = cm(1,1); fp = cm(1,2); fn = cm(2,1); tp = cm(2,2);
    acc  = (tp + tn) / max(1, sum(cm(:)));
    prec = tp / max(1, tp + fp);
    rec  = tp / max(1, tp + fn);
    if (prec+rec)==0, f1 = 0; else, f1 = 2*(prec*rec)/(prec+rec); end
end

function mdl_best = train_svm_optimized_FLF(Xtr, ytr)
% Simple grid search over C and KernelScale with 5 fold CV
    C_list      = [0.25 0.5 1 2 4 8 16 32];
    sigma_list  = [0.5 1 2 4 8 16 32 64];   % KernelScale = sigma

    cv = cvpartition(ytr, 'KFold', 5);
    best_score = -Inf; best_C = 1; best_sigma = 1;

    for c = 1:numel(C_list)
        for s = 1:numel(sigma_list)
            Cval = C_list(c); Sig = sigma_list(s);

            preds = zeros(size(ytr));
            for k = 1:cv.NumTestSets
                tr = training(cv, k);
                te = test(cv, k);
                try
                    mdl = fitcsvm(Xtr(tr,:), ytr(tr), ...
                        'KernelFunction','rbf', ...
                        'BoxConstraint', Cval, ...
                        'KernelScale', Sig, ...
                        'ClassNames', [0 1]);
                catch ME
                    warning('fitcsvm error: %s', ME.message);
                    continue;
                end
                preds(te) = predict(mdl, Xtr(te,:));
            end

            [~, ~, ~, f1] = metrics_from_labels(ytr, preds);
            if f1 > best_score
                best_score = f1; best_C = Cval; best_sigma = Sig;
            end
        end
    end

    mdl_best = fitcsvm(Xtr, ytr, 'KernelFunction','rbf', ...
        'BoxConstraint', best_C, 'KernelScale', best_sigma, ...
        'ClassNames', [0 1]);
end
