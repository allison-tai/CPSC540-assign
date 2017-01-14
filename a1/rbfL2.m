function [model] = rbfL2(X,y,sigma,lambda)
% Compute sizes
[n,d] = size(X);

% Add bias variable
Z = exp(-(X-X').^2/(2*sigma));

% Solve L^2
maxFunEvals = 400; % Maximum number of evaluations of objective
verbose = 1; % Whether or not to display progress of algorithm

w0 = zeros(n,1);

model.X = X;
model.w = findMin(@regLoss,w0,maxFunEvals,verbose,Z,y,lambda);
model.predict = @predict;
end

function [f,g] = regLoss(w,Z,y,lambda)
f = norm(Z*w-y)^2/2+lambda*norm(w)^2/2; % Function value
g = Z'*(Z*w-y)+lambda*w; % Gradient
end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);

Zhat = exp(-(Xhat-model.X').^2/(2));

yhat = Zhat*model.w;
end