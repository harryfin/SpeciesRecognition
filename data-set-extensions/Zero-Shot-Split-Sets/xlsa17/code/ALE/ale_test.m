function [acc] = ale_test(W, X, Y, labels)

classes = unique(labels);
nClass = length(unique(labels));

scores = X * W * Y;
[~, predict_label] = max(scores, [], 2);

acc_per_class = zeros(nClass, 1);
for i = 1 : nClass
    acc_per_class(i) = sum((labels == classes(i)) & (predict_label == classes(i))) / sum(labels == classes(i));
end

acc = mean(acc_per_class);
