function [models label pi] = bernoullimodel(images,labels,ntrain,clusters)

models = []; label = []; pi = [];
for n=0:9

[Xtrain trainlabels] = sampleMNIST(ntrain,images,labels,n);
Xtrain = reshape(Xtrain,[28^2 ntrain])';
model = mixofBernoullis(Xtrain,1/100,clusters);

[samples plots] = model.sample(model,clusters);
plots = plots';

models = cat(4,models,reshape(plots,[28 28 clusters]));
pi = [pi model.pi];
label = [label ones(clusters,1)*n];
end

end

