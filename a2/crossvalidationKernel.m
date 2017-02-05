% Clear variables and close figures
clear all
close all

% Load data
load nonLinear.mat % Loads {X,y,Xtest,ytest}
[n,d] = size(X);
[t,~] = size(Xtest);

% Split training data into a training and a validation set
for i=1:5
    testStart = 1 + n/5*(i-1);
    testEnd = n/5*i;
    trainNdx = [1 : testStart-1 testEnd + 1:n] ;
    %valid = testStart:testEnd;
    valid = datasample(1:n,n/5,'Replace',false); % randomly select 10 validation indices
    Xtraini = X; Xtraini(valid,:) =[]; Xtrain(:,i)=Xtraini; % remove validation from training data
    ytraini = y; ytraini(valid,:) =[]; ytrain(:,i)=ytraini;
    Xvalid(:,i) = X(valid,:);
    yvalid(:,i) = y(valid,:);
end

% Find best value of RBF parameters,
% training on the train set and validating on the validation set
minErr = inf;
for sigma = 1+[-5:0]/60 % range where sigma was found. The random validation selection restrains us from more precision
    for lambda = 2.^[-20:-15] % as above. Note that this is right up against machine precision
        % Train on the training set
        validError=0;
        for i=1:5
            model = kernelRegression(Xtrain(:,i),ytrain(:,i),lambda,sigma);
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
model = kernelRegression(X,y,bestLambda,bestSigma);
fprintf('Optimal sigma: 1+(%1.f/60)\t Optimal lambda: 2^%2.f\nMinimum Error %f\n',(sigma-1)*60,log(lambda)/log(2),minErr);

m = 40;
model1 = subsampling(Xtrain(:,i),ytrain(:,i),lambda,sigma,m);

% Test least squares model on test data
yhat = model.predict(model,Xtest);
yhat1 = model1.predict(model1,Xtest,m);

% Report test error
squaredTestError = sum((yhat-ytest).^2)/t
squaredTestErrorsub = sum((yhat1-ytest).^2)/t


% Plot model
figure(1);
plot(X,y,'b.');
hold on
plot(Xtest,ytest,'g.');
Xhat = [min(X):.1:max(X)]'; % Choose points to evaluate the function
yhat = model.predict(model,Xhat);
yhat1 = model1.predict(model1,Xhat,m);
kern = plot(Xhat,yhat,'r');
sub = plot(Xhat,yhat1,'b');
ylim([-300 400]);
legend([sub kern],'Subsampling','Kernel')