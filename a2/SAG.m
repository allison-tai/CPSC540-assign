clear all
load quantum.mat
[n,d] = size(X);
lambdaFull = 1;

% Initialize
maxPasses = 10;
progTol = 1e-4;
w = zeros(d,1);
lambda = lambdaFull/n; % The regularization parameter on one example
L = .25*max(diag(X'*X)) + lambda;

% Stochastic gradient initialization
gvector = zeros(d,n);
gadd = zeros(d,1);
w_old = w;
for t = 1:maxPasses*n
    % Choose variable to update
    i = randi(n);
    
    % Evaluate the gradient for example i
    [f,g] = logisticL2_loss(w,X(i,:),y(i),lambda);
    
    % Choose the step-size
    alpha = 1/L;
    
    % Take the stochastic gradient step
    if gvector(:,i) ~= zeros(d,1)
        gadd = gadd - gvector(:,i)+g; % replace with 
        gvector(:,i) = g;
    else
        gadd = gadd + g;
        gvector(:,i) = g;
    end 
    w = w - alpha*gadd/n;
    
    if mod(t,n) == 0
        change = norm(w-w_old,inf);
        fprintf('Passes = %d, function = %.4e, change = %.4f\n',t/n,logisticL2_loss(w,X,y,lambdaFull),change);
        if change < progTol
            fprintf('Parameters changed by less than progTol on pass\n');
            break;
        end
        % reset
        gvector = zeros(d,n);
        gadd = zeros(d,1);
        w_old = w;
    end
end