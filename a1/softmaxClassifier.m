function  [model] = softmaxClassifier(X,y)
% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k); % Each column is a classifier
% for c = 1:k
%     yc = ones(n,1); % Treat class 'c' as (+1)
%     yc(y ~= c) = -1; % Treat other classes as (-1)
%     W(:,c) = findMin(@softmaxloss,W(:,c),500,1,X,yc);
% end
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
f = sum(sum(-X'.*W(:,y))+log(sum(exp(W'*X')))); % Function value

g = zeros(d,k);
v = exp(sum(X'.*W(:,y)))./sum(exp(W'*X')); % softmax probability
X2 = sum(X.*(v'*ones(1,d))); % sum(xi * softmax probability for xi)
for c = 1:k
    % only consider the xi where yi == c
    yc = ones(n,d);
    yc(y ~= c,:) = 0;
    g(:,c) = X2 - sum(X.*yc);
end
g = reshape(g, [d*k 1]);
end