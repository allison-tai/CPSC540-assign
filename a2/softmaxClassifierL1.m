function [model] = softmaxClassifier(X,y,lambda)

% Compute sizes
[n,d] = size(X);
k = max(y);

W = zeros(d,k); % Each column is a classifier
%W(:) = findMin(@softmaxLoss,W(:),500,1,X,y,k,lambda);
W(:) = proxGradL1(@(w) softmaxLoss(w,X,y,k),W(:),lambda,500);

model.W = W;
model.predict = @predict;
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end

function [nll,g,H] = softmaxLoss(w,X,y,k)

[n,p] = size(X);
W = reshape(w,[p k]);

XW = X*W;
Z = sum(exp(XW),2);

ind = sub2ind([n k],[1:n]',y);
nll = -sum(XW(ind)-log(Z));% + lambda*norm(W,'fro')^2/2;

g = zeros(p,k);
for c = 1:k
    g(:,c) = X'*(exp(XW(:,c))./Z-(y == c));
end
g = g;% + lambda*W;
g = reshape(g,[p*k 1]);
end