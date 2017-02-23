function [model] = multivariateT(X)

[n,d] = size(X);

% Fit a multivariate student T
mu_old = ones(d,1);
mu = zeros(d,1);
sigma = eye(d);
dof = 3;
fprintf('Fitting multivariate student T density model...\n');
options.Display = 0;
while norm(mu-mu_old,'inf') > 1e-4
    mu_old = mu;
    
    % Update mean
    funObj_mu = @(mu)NLL(X,mu,sigma,dof,1);
    mu = minFunc(funObj_mu,mu,options);
    
    % Update covariance
    funObj_sigma = @(sigma)NLL(X,mu,sigma,dof,2);
    sigma(:) = minFunc(funObj_sigma,sigma(:),options);
    
    % Update dof
    funObj_dof = @(dof)NLL(X,mu,sigma,dof,3);
    dof = minFunc(funObj_dof,dof,options);
end

model.mu = mu;
model.sigma = sigma;
model.dof = dof;
model.pdf = @pdf;

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
        tmp = 1 + (1/dof)*(X(i,:)'-mu)'*sigmaInv*(X(i,:)'-mu);
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
