X = load('statlog.heart.data');
y = X(:,end);
y(y==2) = -1;
X = X(:,1:end-1);
n = size(X,1);

% Add bias and standardize
X = [ones(n,1) standardizeCols(X)];
d = size(X,2);

% Set regularization parameter
lambda = 1;

% Initialize dual variables
z = zeros(n,1);

% Some values used by the dual
YX = diag(y)*X;
G = YX*YX';

% Convert from dual to primal variables
w = (1/lambda)*(YX'*z);

% Evaluate primal objective:
P0 = sum(max(1-y.*(X*w),0)) + (lambda/2)*(w'*w);

e = ones(n,1);
% Evaluate dual objective:
D0 = sum(z) - (z'*G*z)/(2*lambda);
% Dg = e - G*z/lambda % gradient

maxIter = 500;
for j = 1:n*maxIter
    i = randi(n);
    Dg(i) = 1 - G(i,:)*z/lambda;
    %z(i) = z(i)+Dg(i)/sqrt(j);%
    z(i) = Dg(i)*lambda/G(i,i);
    if z(i) > 1
        z(i) = 1;
    elseif z(i) < 0
        z(i) = 0;
    end
end
%z = G\e % gradient = 0 (doesn't work, condition # too high?)
zo = z;

D = sum(z) - (z'*G*z)/(2*lambda);
w = YX'*z/lambda;
P = sum(max(1-y.*(X*w),0)) + (lambda/2)*(w'*w);
fprintf('Dual value: %2.4f\t Primal value: %3.4f\n',D,P)