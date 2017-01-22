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
P = sum(max(1-y.*(X*w),0)) + (lambda/2)*(w'*w)

% Evaluate dual objective:
D = sum(z) - (z'*G*z)/(2*lambda)
