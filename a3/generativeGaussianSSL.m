function [ model ] = generativeGaussianSSL(X, y, Xtilde)

[n,d] = size(X);
k = numel(unique(y));

% initialization of parameters
N = zeros(k,1);
mu = zeros(k,d);
Sigma = zeros(k,d,d);
for c = 1:k
    Xc = X;
    N(c) = sum(y == c);
    % only consider where data is in class c
    Xc(y ~= c,:) = [];
    mu(c,:) = sum(Xc)/N(c);
    Sigma_c = zeros(d,d);
    for i = 1:N(c)
        Sigma_c = Sigma_c + (Xc(i,:)-mu(c,:))'*(Xc(i,:)-mu(c,:));
    end
    Sigma(c,:,:) = Sigma_c/N(c);
end
theta = N/n;

model.k = k;
model.mu = mu;
model.theta = theta;
model.Sigma = Sigma;
model.N = N;

delta = Inf; Lold = -Inf; counter = 0;
while delta>1 % until sufficiently close to maximum
    [model L] = EM(X,Xtilde,y,model);
    delta = L-Lold;
    Lold = L;
    counter = counter + 1;
    fprintf('run %2.f:\tdelta = %1.3f\n',counter,delta)
end

model.predict = @predict;
end

function [yhat] = predict(model, Xhat)
[t, d] = size(Xhat);
k = model.k;
theta = model.theta;
Sigma = model.Sigma;
mu = model.mu;
yhat = zeros(t,1);

% calculate probability of each classification
y_prob = zeros(t,k);
for i = 1:t
    for c = 1:k
        Sigma_c = squeeze(Sigma(c,:,:));
        ldSigma = logdet((Sigma_c));
        y_prob(i,c) = log(theta(c)) - (d/2)*log(2*pi) - 0.5*ldSigma - ...
            0.5*((Xhat(i,:)-mu(c,:))*Sigma_c^(-1)*(Xhat(i,:)-mu(c,:))');
    end
    [M, yhat(i)] = max(y_prob(i,:));
end
end

function [model L] = EM(X,Xtilde,y,model)
    [n, d] = size(X);
    [t, d] = size(Xtilde);
    k = model.k;
    
    N = model.N;
    mu_t = model.mu;
    theta = model.theta;
    Sigma = model.Sigma;
   
    ri = zeros(t,k);
    % E-step: calculate rc^i
    for i = 1:t
        ri_u = zeros(1,k);
        for c = 1:k
            Sigma_c = squeeze(Sigma(c,:,:));
            ri_u(c) = gaussian(Xtilde(i,:),d,Sigma_c,mu_t(c,:),theta(c));
        end
        ri(i,:) = ri_u/sum(ri_u); % soft
        %[M c] = max(ri_u); % hard
        %ri(i,c) = 1; %hard
    end
    r = sum(ri)';
    
    mu = mu_t;
    % M-step
    L = 0;
    for c = 1:k
        Xc = X;
        % only consider where data is in class c
        Xc(y ~= c,:) = [];
        mu(c,:) = (sum(Xc));
        Sigma_c = zeros(d,d);            
        Sigmatc = squeeze(Sigma(c,:,:));
        ldSigma = logdet((Sigmatc));
        Sigmainv = Sigmatc^(-1);
        for i = 1:N(c)
            Sigma_c = Sigma_c + (Xc(i,:)-mu_t(c,:))'*(Xc(i,:)-mu_t(c,:));
            L = L + log(theta(c)) - (d/2)*log(2*pi) - 0.5*ldSigma - ...
                0.5*((Xc(i,:)-mu_t(c,:))*Sigmainv*(Xc(i,:)-mu_t(c,:))');
        end
        for i = 1:t
            Sigma_c = Sigma_c + ri(i,c)*(Xtilde(i,:)-mu_t(c,:))'*(Xtilde(i,:)-mu_t(c,:));
            L = L + ri(i,c)*(log(theta(c)) - (d/2)*log(2*pi) - 0.5*ldSigma - ...
                0.5*((Xtilde(i,:)-mu_t(c,:))*Sigmainv*(Xtilde(i,:)-mu_t(c,:))'));       
        end
        theta(c) = (N(c) + r(c))/(n + t);
        Sigma(c,:,:) = Sigma_c/(N(c) + r(c));
    end
    mu = (mu + ri'*Xtilde)./(N + r);
    
    model.theta = theta;
    model.mu = mu;
    model.Sigma = Sigma;
end