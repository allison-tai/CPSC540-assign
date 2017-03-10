function [ model ] = mixofBernoullis(X,alpha,K)

[n,d] = size(X);

% initialization of parameters
pi = 1;
model.pi = pi/sum(pi);
mu = rand(1,d);
model.mu = 0.1*mu+0.9*rand(K,d);

for i=1:10
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

function [model] = EM(X,model,alpha,K)
    %X = model.X;
    [N D] = size(X);
    pi = model.pi; % P(zi=c|Theta)
    mu = model.mu; % P(xi|zi,Theta)
    % E-step
    Li = log(pi)+log(mu)*X'+log(1-mu)*(1-X');
    %Li = logprod(pi,1)+logprod(mu,X')+logprod(1-mu,1-X');
    if max(max(isnan(Li)))==1
        model.Li=Li
        error('Li = nan')
    end
    logri = Li-log(sum(exp(Li)));
    logri(find(logri==Inf))=-Inf; % correct incidents where Li=0
    model.logri = logri;
    max(max(isnan(Li)==1)); 
    ri = exp(logri);
    % M-step
    Nc = sum(ri,2);
    %mu = (ri*X)./(Nc);
    %mu(find(Nc==0),:)=0;
    %model.mu = (ri*X+1/60000)./(Nc+N/60000);
    model.mu = (ri*X+alpha)./(Nc+N*alpha);
    pi = (Nc+alpha)/sum(Nc+alpha);
    %pi = (Nc+alpha)/sum(Nc+alpha);
    model.pi = pi;
end



    %for n=1:N
    %    zsum = 0;
    %    for k=1:K
    %        if max(mu(k,:))==1 || min(mu(k,:))==0
    %            break
    %        end
    %        logz(n,k)= log(pi(k))+sum(log(mu(k,:)).*(X(n,:))+log(1-mu(k,:)).*(1-X(n,:)));
    %    end
    %    if zsum ~= 0 % get rid of NaN
    %        z(n,:) = z(n,:)/zsum;
    %    end
    %end

    %for i=1:K
     %   if pi(i)~=0
      %      for j=1:D
       %         if mu(i,j)~=0
        %            Li(i,j) = log(pi(i))+log(mu)*X';
         %           logri(i,j) = log(exp(Li))-log(sum(exp(Li)));
          %          ri(i,j) = exp(logri);
           %     end
            %end
        %end