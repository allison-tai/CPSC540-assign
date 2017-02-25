load mnist
Xtrain = Xtrain(1:1000,:);

%model = densityBernoulli(Xtrain,1);
model = mixofBernoullis(Xtrain,1,10);

nlls = model.predict(model,Xtest);
averageNLL = sum(nlls)/size(Xtest,1)

samples = model.sample(model,4);
figure(1);
for i = 1:4
    subplot(2,2,i);
    imagesc(reshape(samples(i,:),[28 28])');
end

%imshow(reshape(Xtrain(randi(60000),:),[28 28]))