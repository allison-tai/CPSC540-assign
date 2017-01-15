function [model] = svRegression(X,y,epsilon)

% Compute sizes
[n,d] = size(X);

% Add bias variable
Z = [ones(n,1) X];

% Solve SVM
f = [ones(n,1); zeros(d+1,1)]; % bias
A = [-eye(n) Z; -eye(n) -Z; -eye(n) zeros(n,d+1)]; % linear program
b = [y+epsilon; -y+epsilon; zeros(n,1)];
x = linprog(f,A,b);

r = x(1:n); w = x(n+1:end);

model.w = w;
model.predict = @predict;

end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);

Zhat = [ones(t,1) Xhat];

yhat = Zhat*model.w;
end