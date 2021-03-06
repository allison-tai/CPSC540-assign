function [model] = generativeStudent(X,Y)
% We'll fit k different multivariate T models
[n, d] = size(X);
k = numel(unique(Y));
mu = zeros(k,d);
dof = zeros(k,1);
sigma = zeros(k,d,d);
for c = 1:k
    Xc = X;
    % only consider where data is in class c
    Xc(Y ~= c,:) = [];
    [model] = multivariateT(Xc);
    mu(c,:) = model.mu;
    dof(c) = model.dof;
    sigma(c,:,:) = model.sigma;
end
model.mu = mu;
model.sigma = sigma;
model.dof = dof;
model.predict = @predict;
end

function [Yhat] = predict(model,Xhat)
[t, d] = size(Xhat);
mu = model.mu;
[k, d] = size(mu);
Y_prob = zeros(k,t); % probability of Y(t) being c

dof = model.dof;
sigma = model.sigma;
sigmac = zeros(d,d);
muc = zeros(d,1);
for c = 1:k
    muc = mu(c,:); model.mu = muc;
    sigmac(:,:) = sigma(c,:,:); model.sigma = sigmac;
    dofc = dof(c); model.dof = dofc;
    [Y_prob(c,:)] = pdf(model,Xhat);
end
[M, Yhat] = max(Y_prob);
end

function [lik] = pdf(model,X)
mu = model.mu;
sigma = model.sigma;
dof = model.dof;

[n,d] = size(X);
nll = zeros(n,1);

[R,err]=chol(sigma);
if err == 0
    sigmaInv = sigma^-1;
    for i = 1:n
        tmp = 1 + (1/dof)*(X(i,:)-mu)*sigmaInv*(X(i,:)-mu)';
        nll(i,1) = ((d+dof)/2)*log(tmp);
    end
    logSqrtDetSigma = sum(log(diag(R)));
    logZ = gammaln((dof+d)/2) - (d/2)*log(pi) - logSqrtDetSigma - gammaln(dof/2) - (d/2)*log(dof);
    nll = nll - logZ;
    lik = exp(-nll);
else
    lik(:) = inf;
end
end

function [nll,g] = NLL(X,mu,sigma,dof,deriv)
[n,d] = size(X);

nll = 0;
switch deriv
    case 1
        g = zeros(d,1); % derivative wrt mu
    case 2
        % derivative wrt sigma
        g = zeros(d);
    case 3
        % derivative wrt dof
        g = 0;
end

if length(sigma) ~= d
    sigma = reshape(sigma,d,d);
    sigma = (sigma+sigma')/2;
end
[R,err]=chol(sigma);
if err == 0
    sigmaInv = sigma^-1;
    for i = 1:n
        tmp = 1 + (1/dof)*(X(i,:)'-mu)'*sigmaInv*(X(i,:)'-mu);
        nll = nll + ((d+dof)/2)*log(tmp);
        
        switch deriv
            case 1
                g = g - ((d+dof)/(dof*tmp))*sigmaInv*(X(i,:)'-mu);
            case 2
                g = g - ((d+dof)/(2*dof*tmp))*sigmaInv*(X(i,:)'-mu)*(X(i,:)'-mu)'*sigmaInv;
            case 3
                g = g - (d/(2*tmp*dof^2))*(X(i,:)'-mu)'*sigmaInv*(X(i,:)'-mu);
                g = g - (dof/(2*tmp*dof^2))*(X(i,:)'-mu)'*sigmaInv*(X(i,:)'-mu);
                g = g + (1/2)*log(tmp);
        end
    end
else
    nll = inf;
    g = g(:);
    return
end

% Now take into account logZ
logSqrtDetSigma = sum(log(diag(R)));
logZ = gammaln((dof+d)/2) - (d/2)*log(pi) - logSqrtDetSigma - gammaln(dof/2) - (d/2)*log(dof);
nll = nll - n*logZ;
switch deriv
    case 2
        g = g + (n/2)*sigmaInv;
        g = (g+g')/2;
        g = g(:);
    case 3
        g = g - (n/2)*psi((dof+d)/2) + (n/2)*psi(dof/2) + n*(d/(2*dof));
end
end