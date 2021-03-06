function [model] = softmaxClassifierGL1(X,y,lambda)

% Compute sizes
[n,d] = size(X);
k = max(y);
Groups = (1:100)'*ones(1,k); % enumerating each row as a group


W = zeros(d,k);
Groups = reshape(Groups,[d*k 1]); % reshape group enumeration (to be eventually compatible with w)

% Each column is a classifier
W(:) = proxGradGroupL1(@(w) softmaxLoss(w,X,y,k,count),W(:),Groups,lambda,500);

model.W = W;
model.predict = @predict;
end

function [yhat] = predict(model,X)
W = model.W;
[~,yhat] = max(X*W,[],2);
end

function [nll,g,H] = softmaxLoss(w,X,y,k,count)

[n,p] = size(X);
W = reshape(w,[p k]);

XW = X*W;
Z = sum(exp(XW),2);

ind = sub2ind([n k],[1:n]',y);
nll = -sum(XW(ind)-log(Z));

g = zeros(p,k);
for c = 1:k
    g(:,c) = X'*(exp(XW(:,c))./Z-(y == c));
end
g = g;
g = reshape(g,[p*k 1]);
end