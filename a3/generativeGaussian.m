function [model] = generativeGaussian(X,Y)
[n, d] = size(X);
k = numel(unique(Y));
N = zeros(k,1);
mu = zeros(k,d);
Sigma = zeros(k,d,d);
% basically, derive mu_c and sigma_c
for c = 1:k
    Xc = X;
    N(c) = sum(Y == c);
    % only consider where data is in class c
    Xc(Y ~= c,:) = [];
    mu(c,:) = sum(Xc)/N(c);
    Sigma_c = zeros(d,d);
    for i = 1:N(c)
        Sigma_c = Sigma_c + (Xc(i,:)-mu(c))'*(Xc(i,:)-mu(c));
    end
    Sigma(c,:,:) = Sigma_c/N(c);
end
theta = N/n;
model.k = k;
model.theta = theta;
model.mu = mu;
model.Sigma = Sigma;
model.predict = @predict;
end

function [Yhat] = predict(model,Xhat)
% yhat_c = theta_c * p(x|y = c, theta), where part 2 is the gaussian
% distribution using our mu_c/sigma_c
% then yhat is max of all yhat_c
[t, d] = size(Xhat);
k = model.k;
theta = model.theta;
Sigma = model.Sigma;
mu = model.mu;
Yhat = zeros(t,1);

% calculate probability of each classification
Y_prob = zeros(t,k);
for i = 1:t
    for c = 1:k
        Sigma_c = squeeze(Sigma(c,:,:));
        ldSigma = logdet((Sigma_c));
        Y_prob(i,c) = log(theta(c)) - (d/2)*log(2*pi) - 0.5*ldSigma - ...
            0.5*((Xhat(i,:)-mu(c))*inv(Sigma_c)*(Xhat(i,:)-mu(c))');
    end
    [M, Yhat(i)] = max(Y_prob(i,:));
end
end