function w = AdaGrad(X,y,delta)
[n,d] = size(X);
lambdaFull = 1;

% Initialize
maxPasses = 1;
progTol = 1e-4;
w = zeros(d,1);
lambda = lambdaFull/n; % The regularization parameter on one example
w_old = w;

% initialize D
normD = delta*ones(d,1); % D(i,i) denomonator
D = spdiags(ones(d,1)./(sqrt(normD)),0,d,d);

% Stochastic gradient
for t = 1:maxPasses*n
    % Choose variable to update
    i = randi(n);
    
    % Evaluate the gradient for example i
    [f,g] = logisticL2_loss(w,X(i,:),y(i),lambda);
    
    % Choose the step-size
    alpha = 1/(lambda*t);%(t-t^(0.5)+1));
    
    % construct D
    for i=1:d
        if g(i) ~= 0
            normD(i) = normD(i) + g(i)^2; % update the sum
            D(i,i) = 1/(sqrt(normD(i))); % update the diagonal
        end
    end
    
    % Take the stochastic gradient step
    w = w - alpha*D*g;
    
    if mod(t,n) == 0
        change = norm(w-w_old,inf);
        fprintf('Passes = %d, function = %.4e, change = %.4f\n',t/n,logisticL2_loss(w,X,y,lambdaFull),change);
        if change < progTol
            fprintf('Parameters changed by less than progTol on pass\n');
            break;
        end
        w_old = w;
    end
end
end