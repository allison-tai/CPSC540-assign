
% Clear variables and close figures
clear all
close all

% Load data
load nonLinear.mat % Loads {X,y,Xtest,ytest}
[n,d] = size(X);
[t,~] = size(Xtest);

% Train least squares model on training data
%sigma=1;lambda=1;
%model = leastSquaresRBFL2(X,y,sigma,lambda);

% Split training data into a training and a validation set
for i=1:10
    valid = (i-1)*10+[1:10];
    %valid = datasample(1:n,10,'Replace',false); % randomly select 10 validation indices
    Xtraini = X; Xtraini(valid,:) =[]; Xtrain(:,i)=Xtraini; % remove validation from training data
    ytraini = y; ytraini(valid,:) =[]; ytrain(:,i)=ytraini;
    Xvalid(:,i) = X(valid,:);
    yvalid(:,i) = y(valid,:);
end

% Find best value of RBF kernel parameter,
%   training on the train set and validating on the validation set
minErr = inf;
for sigma = 1+[-40:-20]/60
    for lambda = 0
        % Train on the training set
        validError=0;
        for i=1:10
            model = leastSquaresRBFL2(Xtrain(:,i),ytrain(:,i),sigma,lambda);
    
            % Compute the error on the validation set
            yhat = model.predict(model,Xvalid(:,i));
            validError = validError+sum((yhat - yvalid(:,i)).^2)/(n/10);
        end
        % Keep track of the lowest validation error
        if validError < minErr
            minErr = validError;
            bestSigma = sigma;
            bestLambda = lambda;
        end
    end
end
sigma=bestSigma; lambda=bestLambda;
model = leastSquaresRBFL2(X,y,bestSigma,bestLambda);
fprintf('Optimal sigma: 1+%f/60\nOptimal lambda: 2^%f\n',(sigma-1)*60,log(lambda)/log(2));

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