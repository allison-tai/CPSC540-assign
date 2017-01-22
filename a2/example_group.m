clear all

load groupData.mat

model = softmaxClassifier(X,y);

yhat = model.predict(model,X);
errTrain = sum(yhat~=y)/length(y)

yhat = model.predict(model,Xvalid);
errTest = sum(yhat~=yvalid)/length(yvalid)
nModelParams = nnz(model.W)
nFeaturesUsed = nnz(sum(abs(model.W),2))
imagesc(model.W);