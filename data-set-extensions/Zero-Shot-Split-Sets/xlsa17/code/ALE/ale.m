function ale(dataroot, dataset, image_embedding_str, class_embedding_str, eta, nepoch, val, output)

%rng('shuffle');
%% Set the parameters
param.eta = str2double(eta);
param.nepoch = str2num(nepoch);

%% loading data
load([dataroot '/' dataset '/' image_embedding_str '.mat'], 'features', 'labels');
load([dataroot '/' dataset '/' class_embedding_str  '_splits.mat']);
features = features';

if(val == '1')
    disp('Validation using unseen classes...');
    X = features(train_loc, :);
    L = labels(train_loc, :);
    trainclasses = unique(L);
    L = mapLabel(L, trainclasses);
    Y_seen = att(:, trainclasses);

    Xtest = features(val_loc, :);
    Ltest = labels(val_loc, :);
    valclasses = unique(Ltest);
    Ltest = mapLabel(Ltest, valclasses);
    Y_unseen = att(:, valclasses);

    X = [X; train_unseen_X];
    L = [L;train_unseen_labels+size(Y_seen,2)];
    output = [output '_unseen_val'];
else
    disp('Test on unseen classes...');
    X = features(trainval_loc, :);
    L = labels(trainval_loc, :);
    trainvalclasses = unique(L);
    L = mapLabel(L, trainvalclasses);
    Y_seen = att(:, trainvalclasses);

    Xtest = features(test_unseen_loc, :);
    Ltest = labels(test_unseen_loc, :);
    testclasses = unique(Ltest);
    Ltest = mapLabel(Ltest, testclasses);
    Y_unseen = att(:, testclasses);

    X = [X; train_unseen_X];
    L = [L;train_unseen_labels+size(Y_seen,2)];
end
[X, xtest_mean, xtest_variance, xtest_max] = normalization(X);
Xtest = normalization(Xtest, xtest_mean, xtest_variance, xtest_max);

clear features labels;

    


%% Start to train the model
disp([image_embedding_str ' ' class_embedding_str]);
disp([dataset ', eta=' num2str(param.eta) ', nepoch=' num2str(param.nepoch)]);

%% Initialization

W = 1.0/sqrt(size(X, 2)) * randn(size(X, 2), size(Y, 1));

for i=1:param.nepoch
    W = ale_train(W, X, L, Y, param.eta);
    acc = ale_test(W, Xtest, Ytest, Ltest);
    disp(['Epoch ' num2str(i) ', top-1 accuracy=' num2str(acc(1)) ]);
end

% Save the results
f = fopen([output '.txt'], 'w');
msg = [num2str(acc(1)) ' ' eta ' ' nepoch];
fprintf(f, '%s\n', msg);
fclose(f);

save([output '.mat'], 'W', 'acc');
