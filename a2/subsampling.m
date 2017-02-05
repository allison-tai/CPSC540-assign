function [model] = subsampling(X,y,lambda,sigma,m)

% Compute sizes
[n,d] = size(X);

% Add bias variable
K = rbfBasis(X,X,sigma);
K11 = K(1:m,1:m);
K12 = K(1:m,m+1:n);
K21 = K(m+1:n,1:m);
K22 = K(m+1:n,m+1:n);
K1 = [K11;K21];

% Solve least squares problem
z = (K1'*K1 + lambda*K11)\(K1'*y);

model.X = X;
model.z = z;
model.sigma = sigma;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat,m)
[t,d] = size(Xhat);

Khat = rbfBasis(Xhat,model.X,model.sigma);
Khat1 = Khat(:,1:m);

yhat = Khat1*model.z;
end

function [Xrbf] = rbfBasis(X1,X2,sigma)
n1 = size(X1,1);
n2 = size(X2,1);
d = size(X1,2);
Z = 1/sqrt(2*pi*sigma^2);
D = X1.^2*ones(d,n2) + ones(n1,d)*(X2').^2 - 2*X1*X2';
Xrbf = Z*exp(-D/(2*sigma^2));
end