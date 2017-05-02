function [models label pi] = bernoullimodel(images,labels,ntrain,clusters)

models = []; label = []; pi = [];
for n=0:9

[Xtrain trainlabels] = sampleMNIST(ntrain,images,labels,n);
Xtrain = reshape(Xtrain,[28^2 ntrain])';
model = mixofBernoullis(Xtrain,1/100,clusters+1);

[samples plots] = model.sample(model,clusters+1);
plots = plots(1:clusters,:)';
model.pi = sort(model.pi(:),'descend');
model.pi = model.pi(1:clusters)/sum(model.pi(1:clusters));

models = cat(3,models,reshape(plots,[28 28 clusters]));
pi = [pi model.pi(1:clusters)];
label = [label ones(clusters,1)*n];
end

end

