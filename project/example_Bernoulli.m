load MNIST

models = [];
for n=0:9
clusters = 5;
ntrain = 500; ntest = 1;
[Xtrain trainlabels] = sampleMNIST(ntrain,images,labels,n);
Xtest = images(:,:,randi(length(labels),[ntest 1]));
Xtrain = reshape(Xtrain,[28^2 ntrain])';
Xtest = reshape(Xtest,[28^2 ntest])';
model = mixofBernoullis(Xtrain,1/100,clusters);

nlls = model.predict(model,Xtest);
averageNLL = sum(nlls)/size(Xtest,1)

if isnan(averageNLL)==1 % stop opening figures
    error('averageNLL == NaN')
end

[samples plots] = model.sample(model,clusters);
figure(2)
for i = 1:clusters
    subplot(3,3,i);
    imagesc(reshape(plots(i,:),[28 28]));
end
plots = plots';
models = cat(4,models,reshape(plots,[28 28 clusters]));
pause
end
return

figure(1);
for i = 1:9
    subplot(3,3,i);
    imagesc(reshape(samples(i,:),[28 28])');
end
