function [model] = leastSquaresEmpiricalBayes(X, y)

% param search
bestDegree = 1;
bestSigma = 1;
bestLambda = 1;
bestNLL = inf;
for degree = 1:10
    for sigma = logspace(-5,5,11)
        for lambda = logspace(-5,5,11)
            nll = NLL(degree,X,y,sigma,lambda);
            if nll < bestNLL
                bestNLL = nll;
                bestLambda = lambda;
                bestSigma = sigma;
                bestDegree = degree;
            end
        end
    end
end

% construct basis
Xpoly = polyBasis(X, bestDegree);
[n,d] = size(Xpoly);
XSig = Xpoly'*(eye(n)/bestSigma^2);

% Solve for w
w = (XSig*Xpoly + bestLambda*eye(d))\XSig*y;

model.w = w;
model.lambda = bestLambda;
model.sigma = bestSigma;
model.degree = bestDegree;
model.predict = @predict;
end

function [yhat] = predict(model, Xtest)
Xpoly = polyBasis(Xtest,model.degree);
yhat = Xpoly*model.w;
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
nll = logdet(C, inf) + y'*v;
end