function [model] = leastSquaresEmpiricalBayes(X, y)

% grid search
degree = linspace(1,10,10);
sigma = [1, 10, 100, 1000, 1e4, 1e5];
lambda = [1, 10, 100, 1000, 1e4, 1e5];
[D, S, L] = ndgrid(degree, sigma, lambda);

nll = arrayfun(@(p1,p2,p3) NLL(p1,X,y,p2,p3), D, S, L); 
[minval, minidx] = min(nll);
bestDegree = D(minidx);
bestSigma = S(minidx);
bestLambda = L(minidx);

% construct basis
Xpoly = polyBasis(X, bestDegree);

% Calculate w using Empirical Bayes
w = (Xpoly'*Xpoly)\Xpoly'*y;

model.w = w;
model.lambda = bestLambda;
model.sigma = bestSigma;
model.degree = bestDegree;
model.predict = @predict;
end

function [Xpoly] = polyBasis(X,m)
n = length(X);
Xpoly = zeros(n,m+1);
for i = 0:m
    Xpoly(:,i+1) = X.^i;
end
end

function [nll] = NLL(m, X, y, sigma, lambda)
[n, d] = size(X);
% construct basis
Xpoly = polyBasis(X, m);
% cost function
C = eye(n)/(sigma^2) + (Xpoly*Xpoly')/lambda;
% calculate NLL
v = C\y;
nll = logdet(C) + y'*v;
end