function [model] = kitchensink(X,y,lambda,sigma,m)

% Compute sizes
[n,d] = size(X);

% Sample from Gaussian
R = sqrt(sigma)*randn(d,m);

% Add bias variable
Z = exp(1i*X*R);

% Solve least squares problem
w = (Z'*Z + lambda)\(Z'*y);

model.R = R;
model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);

Zhat = exp(1i*Xhat*model.R);

yhat = Zhat*model.w;
end