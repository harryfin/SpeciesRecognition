function evaluate(score, dataset, model_path)
%
% Evaluation code of our benchmark in  
% Y. Xian,  B. Schiele, Z. Akata. 
% Zero-shot Learning - The Good, the Bad and the Ugly. IEEE CVPR 2017.
% Cite the above paper if you are using this code.
%
% Without loss of generality, we take the prediction function of any
% zero-shot learning method as argmax_{y \in Y}score(x, a_y), where
% x is an instance, a_y is the attribute vector of class y, score(x,a_y) 
% measures the compatibility score between x and a_y. Thus the prediction
% function searches a class with the highest score in classes set Y. 
%
% In zero-shot learning (zsl), Y takes the set of all unseen classes
% In generalized zero-shot learning (gzsl), Y takes the set of all classes
% (including both seen and unseen classes) 
% 
%
% Usage: evaluate(score, dataset)
%
% To use this function, please train your model and write a score function 
% for the method you want to evaluate. The score function should be defined 
% as follow.
% 
% S = score(X, att, model_path)
%
% X:            test instances organized in row
% att:          attribute vectors of your interested classes (in zsl, they 
%               are unseen classses, in gzsl, they are all classes)
% model_path:   path of your pretrained model
% S:            return scores of all instances for each interested classses
%               instances are orgainized in row
%
% Inputs:
%   score:      score function handler of any zsl method                    
%   dataset:    name of the dataset
%   model_path:   path of your pretrained model
%
%
% Outputs:
%   display the averaged per-class accuracies in both zero-shot and 
%   generalizled zero-shot learning
%
%
% Yongqin Xian
% e-mail: yxian@mpi-inf.mpg.de
% Computer Vision and Multimodal Computing, Max Planck Institute Informatics
% Saarbruecken, Germany
% http://d2.mpi-inf.mpg.de
%
%
%% loading data
load(['../data/' dataset '/att_splits.mat'], 'att', 'test_unseen_loc', 'test_seen_loc');
load(['../data/' dataset '/res101.mat'], 'features', 'labels');

test_unseen_X = features(:, test_unseen_loc)';
test_unseen_labels = labels(test_unseen_loc, :);
test_seen_X = features(:, test_seen_loc)';
test_seen_labels = labels(test_seen_loc, :);

unseenclasses = unique(test_unseen_labels);
seenclasses = unique(test_seen_labels);


%% ZSL
zsl_unseen_S = score(test_unseen_X,  att(:, unseenclasses), model_path);
[~, predict_label] = max(zsl_unseen_S, [], 2);
zsl_unseen_predict_label = mapLabel(predict_label, unseenclasses);
zsl_unseen_acc = computeAcc(zsl_unseen_predict_label, test_unseen_labels, unseenclasses);

disp(['ZSL: averaged per-class accuracy=' num2str(zsl_unseen_acc) ' on ' dataset]);

%% GZSL
gzsl_unseen_S = score(test_unseen_X, att, model_path);
[~, gzsl_unseen_predict_label] = max(gzsl_unseen_S, [], 2);
gzsl_unseen_acc = computeAcc(gzsl_unseen_predict_label, test_unseen_labels, unseenclasses);

gzsl_seen_S = score(test_seen_X, att, model_path);
[~, gzsl_seen_predict_label] = max(gzsl_seen_S, [], 2);
gzsl_seen_acc = computeAcc(gzsl_seen_predict_label, test_seen_labels, seenclasses);

H = 2 * gzsl_unseen_acc * gzsl_seen_acc / (gzsl_unseen_acc + gzsl_seen_acc);

disp(['GZSL unseen: averaged per-class accuracy=' num2str(gzsl_unseen_acc) ' on ' dataset]);
disp(['GZSL seen: averaged per-class accuracy=' num2str(gzsl_seen_acc) ' on ' dataset]);
disp(['GZSL: H=' num2str(H) ' on ' dataset]);


function mappedL = mapLabel(L, classes)
    mappedL = -1 * ones(size(L));
    for i=1:length(classes)
        mappedL(L == i) = classes(i);
    end

    

function acc_per_class = computeAcc(predict_label, true_label, classes) 
    nclass = length(classes);
    acc_per_class = zeros(nclass, 1);
    for i=1:nclass
        idx = find(true_label==classes(i));
        acc_per_class(i) = sum(true_label(idx) == predict_label(idx)) / length(idx);
    end
    acc_per_class = mean(acc_per_class);

    
