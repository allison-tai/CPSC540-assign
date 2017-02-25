%clear all

load groupData.mat;
lambda = 10;

%model = softmaxClassifier(X,y);
model = softmaxClassifierL2(X,y,lambda);
%model = softmaxClassifierL1(X,y,lambda);
%model = softmaxClassifierGL1(X,y,lambda);


yhat = model.predict(model,X);
errTrain = sum(yhat~=y)/length(y);

yhat = model.predict(model,Xvalid);
errTest = sum(yhat~=yvalid)/length(yvalid);
nModelParams = nnz(model.W);
nFeaturesUsed = nnz(sum(abs(model.W),2));
fprintf('Training Error: %1.4f\t Test Error: %1.4f\t Parameters: %3.f\t Features: %2.f\n',errTrain,errTest,nModelParams,nFeaturesUsed)
imagesc(model.W);
