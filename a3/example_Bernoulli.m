load mnist
%Xtrain = Xtrain(1:18000,:);

%model = densityBernoulli(Xtrain,1);
model = mixofBernoullis(Xtrain,1/100,10);

nlls = model.predict(model,Xtest);
averageNLL = sum(nlls)/size(Xtest,1)

if isnan(averageNLL)==1 % stop opening figures
    error('averageNLL == NaN')
end

[samples plots] = model.sample(model,9);
figure(1);
for i = 1:4
    subplot(2,2,i);
    imagesc(reshape(samples(i,:),[28 28])');
end
figure(2)
for i = 1:9
    subplot(3,3,i);
    imagesc(reshape(plots(i,:),[28 28])');
end
