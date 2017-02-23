%%
close all;
clear;

load gaussNoise.mat

[n, d] = size(Xtrain);
nClasses = numel(unique(Ytrain));

model = KNN(Xtrain, Ytrain, 1);
Yhat = model.predict(model, Xtest);
fprintf('KNN accuracy is %.2f\n', mean(Yhat==Ytest));

%model = generativeGaussian(Xtrain, Ytrain);
%Yhat = model.predict(model, Xtest);
%fprintf('Gaussian Gen. Model. accuracy is %.2f\n', mean(Yhat==Ytest));

% model = generativeStudent(Xtrain, Ytrain);
% Yhat = model.predict(model, Xtest);
% fprintf('Tdist Gen. Model. accuracy is %.2f\n', mean(Yhat==Ytest));
