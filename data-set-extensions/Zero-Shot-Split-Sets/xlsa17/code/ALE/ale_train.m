function [W] = ale_train(W, X, labels, Y, eta)

n_train = size(X,1);
n_class = size(Y,2);
perm = randperm(n_train);

for i = 1:n_train
    ni = perm(i);
    rand_y = labels(ni);
    yi = labels(ni);
    count = 1;
    scores = X(ni, :) * W * Y;
    for j=1:n_class
        rand_y = labels(ni);
        while(rand_y==yi)
            rand_y =  randi(n_class);
        end
        if(scores(rand_y) + 0.1 > scores(yi))
            W = W - (1/count) * eta * X(ni,:)' * (Y(:,rand_y) - Y(:,labels(ni)))';
            %if count > 1
                %disp(['Ite ' num2str(i) ': count=' num2str(count)]);
            %end
            break;
        end
        count = count + 1;
    end
end