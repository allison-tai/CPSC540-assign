function [model] = regularizedLogisticRegression(X,y,lambda)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];

% Initial values of regression parameters
w = zeros(d+1,1);

% Solve logistic regression problem
maxIter = 500;
verbose = 1;
w = findMin_new(@objective,w,maxIter,verbose,X,y,lambda);

model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end

function [nll,g,h] = objective(w,X,y,lambda)
yXw = y.*(X*w);

% Function value
nll = sum(log(1+exp(-yXw))) + (lambda/2)*(w'*w);

% Gradient
sigmoid = 1./(1+exp(-yXw));
g = -X'*(y.*(1-sigmoid)) + lambda*w;

% Hessian
h = X'*(diag(y.^2.*(sigmoid.^2.*exp(-yXw))))*X + lambda*eye(3);
end