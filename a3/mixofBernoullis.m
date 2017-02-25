function [ model ] = mixofBernoullis(X,alpha,K)

[n,d] = size(X);

theta = sum(X+alpha)/sum(sum(X+alpha));

pi = rand(K,1);
model.pi = pi/sum(pi);
model.mu = rand(K,d);
for i=1
    model = EM(X,model,alpha,K);
end
model.predict = @predict;
model.sample = @sample;
end

function nlls = predict(model, Xhat)
[t,d] = size(Xhat);
pi = model.pi;
mu = model.mu;

nlls = -sum(prod0(Xhat,repmat(log(pi'*mu),[t 1])) + prod0(1-Xhat,repmat(log(1-pi'*mu),[t 1])),2);
end

function samples = sample(model,t)
mu = model.mu;
d = length(mu(t,:));

samples = zeros(t,d);
for i = 1:t
    samples(i,:) = rand(1,d) < mu(t,:);
end
end

function [model] = EM(X,model,alpha,K)
    %X = model.X;
    [N D] = size(X);
    pi = model.pi;
    mu = model.mu;
    z = zeros(N,K);
    for n=1:N
        zsum = 0;
        for k=1:K
            z(n,k)= pi(k)*prod(mu(k,:).^(X(n,:)).*(1-mu(k,:)).^(1-X(n,:)));
            zsum = zsum + z(n,k);
        end
        z(n,:) = z(n,:)/zsum;
    end
    Nm = sum(z);
    mu = z'*X;
    pi = Nm/sum(Nm);
end


