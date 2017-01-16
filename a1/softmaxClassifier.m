function  [model] = softmaxClassifier(X,y)
% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k); % Each column is a classifier
W(:) = findMin(@softmaxLoss,W(:),500,1,X,y,k);
d = derivativeCheck(@softmaxLoss,W(:),X,y,k);

model.W = W;
model.predict = @predict;
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end

function [f,g] = softmaxLoss(w,X,y,k)
[n, d] = size(X);
W = reshape(w, [d k]);
v = sum(exp(W'*X')); % recurring value
f = sum(sum(-X'.*W(:,y))+log(v)); % Function value

g = zeros(d,k);
for c = 1:k
    % only consider the xi where yi == c
    vc = exp(X*W(:,c))./v';
    X2 = sum(X.*(vc*ones(1,d)));
    yc = ones(n,d);
    yc(y ~= c,:) = 0;
    g(:,c) = X2 - sum(X.*yc);
end
g = reshape(g, [d*k 1]);
end