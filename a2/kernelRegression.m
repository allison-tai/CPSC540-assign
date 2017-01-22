function [model] = kernelRegression(X,y,lambda,sigma)

% Compute sizes
[n,d] = size(X);

% Add bias variable
K = rbfBasis(X,X,sigma);

% Solve least squares problem
z = (K' + lambda*eye(n))\y;

model.X = X;
model.z = z;
model.sigma = sigma;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);

Khat = rbfBasis(Xhat,model.X,model.sigma);

yhat = Khat*model.z;
end

function [Xrbf] = rbfBasis(X1,X2,sigma)
n1 = size(X1,1);
n2 = size(X2,1);
d = size(X1,2);
Z = 1/sqrt(2*pi*sigma^2);
D = X1.^2*ones(d,n2) + ones(n1,d)*(X2').^2 - 2*X1*X2';
Xrbf = Z*exp(-D/(2*sigma^2));
end