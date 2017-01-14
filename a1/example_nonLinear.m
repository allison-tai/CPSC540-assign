% DONE 
% Clear variables and close figures
clear all
close all

% Load data
load nonLinear.mat % Loads {X,y,Xtest,ytest}
[n,d] = size(X);
[t,~] = size(Xtest);

% Train least squares model on training data
sigma=0.5;lambda=1;
model = leastSquaresRBFL2(X,y,sigma,lambda);

% Test least squares model on test data
yhat = model.predict(model,Xtest);

% Report test error
squaredTestError = sum((yhat-ytest).^2)/t

% Plot model
figure(1);
plot(X,y,'b.');
hold on
plot(Xtest,ytest,'g.');
Xhat = [min(X):.1:max(X)]'; % Choose points to evaluate the function
yhat = model.predict(model,Xhat);
plot(Xhat,yhat,'r');
ylim([-300 400]);