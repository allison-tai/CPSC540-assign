
load logisticData.mat
tic
%model = logisticL2(X,y,1); % Runs out of iterations... Not sure if this is ok...
[model steps] = logisticL2rand(X,y,1);
toc