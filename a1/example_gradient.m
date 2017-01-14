%load binaryLinear.mat
load rcv1_train_binary.mat

lambda = 1;
%model = logisticL2h(X,y,lambda); % question 3.1
model = logisticL2(X,y,lambda);

binaryClassifier2Dplot(X,y,model);