
load SSL.mat

model = KNN(X, y, 1);
yhat = model.predict(model, Xtest);
fprintf('KNN accuracy is %.2f\n', mean(yhat==ytest));

model = generativeGaussian(X, y);
yhat = model.predict(model, Xtest);
fprintf('Gaussian Gen. Model. accuracy is %.2f\n', mean(yhat==ytest));

 model = generativeGaussianSSL(X, y, Xtilde);
 yhat = model.predict(model, Xtest);
 fprintf('SSL Gauss. Gen. Model. accuracy is %.2f\n', mean(yhat==ytest));
