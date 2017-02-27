function [model] = generativeStudent(X,Y)
% We'll fit k different multivariate T models
[n, d] = size(X);
k = numel(unique(Y));
model = zeros(k);
for c = 1:k
    Xc(Y ~= c,:) = [];
    model(c) = multivariate(Xc);
end

function [Yhat] = predict(model,Xhat)
    
end