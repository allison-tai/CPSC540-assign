clear all
close all

%% Density estimation with multivariate student T

% Generate data from a Gaussian with outliers
nInstances = 250;
nVars = 2;
nOutliers = 25;
mu = randn(nVars,1);
sigma = randn(nVars);
sigma = sigma+sigma'; % Make symmetric
sigma = sigma + (1-min(eig(sigma)))*eye(nVars);
X = mvnrnd(mu,sigma,nInstances);
X(ceil(rand(nOutliers,1)*nInstances),:) = abs(10*rand(nOutliers,nVars));

model = multivariateT(X);

% Plot data and densities
figure;
plot(X(:,1),X(:,2),'.');
increment = 100;
domain1 = xlim;
domain1 = domain1(1):(domain1(2)-domain1(1))/increment:domain1(2);
domain2 = ylim;
domain2 = domain2(1):(domain2(2)-domain2(1))/increment:domain2(2);
d1 = repmat(domain1',[1 length(domain1)]);
d2 = repmat(domain2,[length(domain2) 1]);
lik_T = model.pdf(model,[d1(:) d2(:)]);
contour(d1,d2,reshape(lik_T,size(d1)));hold on;
plot(X(:,1),X(:,2),'.');
title('Data and Density under Multivariate T');
