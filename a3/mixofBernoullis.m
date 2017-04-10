function [ model ] = mixofBernoullis(X,alpha,K)

[n,d] = size(X);

% initialization of parameters
model.pi = ones(K,1)/K;
mu = rand(1,d);
model.mu = 0.1*mu+0.9*X(randi(n,K,1),:);

delta = Inf; Lold = -Inf; counter = 0;
while delta>10000 % until sufficiently close to maximum
    [model L] = EM(X,model,alpha,K);
    delta = L-Lold;
    Lold = L;
    counter = counter + 1;
    fprintf('run %2.f:\tdelta = %1.3f\n',counter,delta)
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

function [samples plots] = sample(model,t)
mu = model.mu;
pi = model.pi;
d = length(mu(t,:));

samples = zeros(t,d);
% pick samples from distributions with largest pi
[sortedValues,sortIndex] = sort(pi(:),'descend');
maxIndex = sortIndex(1:t); 
for i = 1:t
    samples(i,:) = rand(1,d) < mu(maxIndex(i),:);
    plots(i,:) = mu(maxIndex(i),:);
end
end

function [model L] = EM(X,model,alpha,K)
    [N D] = size(X);
    pi = model.pi; % P(zi=c|Theta)
    mu = model.mu; % P(xi|zi,Theta)
    % E-step
    Li = log(pi)+log(mu)*X'+log(1-mu)*(1-X');
    logri = Li-log(sum(exp(Li)));
    model.logri = logri;
    max(max(isnan(Li)==1)); 
    ri = exp(logri);
    L = sum(sum(ri.*Li));
    % M-step
    Nc = sum(ri,2);
    %mu = (ri*X)./(Nc);
    %mu(find(Nc==0),:)=0;
    %model.mu = (ri*X+1/60000)./(Nc+N/60000);
    model.mu = (ri*X+alpha)./(Nc+N*alpha);
    pi = (Nc)/sum(Nc);
    %pi = (Nc+alpha)/sum(Nc+alpha);
    model.pi = pi;
end


