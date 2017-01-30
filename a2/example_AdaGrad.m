load quantum.mat

lambda = 1;
delta = 1;
w = AdaGrad(X,y,delta);
%fprintf('function = %.4e \n',logisticL2_loss(w,X,y,lambda));
