
% Clear variables and close figures
clear all
close all

% Load data
load basisData.mat % Loads X and y
[n,d] = size(X);

% Fit least-squares model
% degree = 2;
% model = leastSquaresBasis(X,y,degree);

% Fit least-squares empirical Bayes model
model = leastSquaresEmpiricalBayes(X,y);

% Print best values
fprintf('sigma = %d\n', model.sigma);
fprintf('lambda = %d\n', model.lambda);
fprintf('degree = %d\n', model.degree);

% Compute training error
yhat = model.predict(model,X);
trainError = sum((yhat - y).^2)/n;
fprintf('Training error = %.2f\n',trainError);

% Compute test error
t = size(Xtest,1);
yhat = model.predict(model,Xtest);
testError = sum((yhat - ytest).^2)/t;
fprintf('Test error = %.2f\n',testError);

% Plot model
figure(1);
plot(X,y,'b.');
title('Training Data');
hold on
Xhat = [min(X):.1:max(X)]'; % Choose points to evaluate the function
yhat = model.predict(model,Xhat);
plot(Xhat,yhat,'g');
