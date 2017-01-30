load quantum.mat
[n,d] = size(X);
lambdaFull = 1;

% Initialize
maxPasses = 10;
progTol = 1e-4;
w = zeros(d,1);
lambda = lambdaFull/n; % The regularization parameter on one example

% Stochastic gradient
w_old = w;
delta = 5; % parameter
garray = [];
wrecord = w_old;
pcent = 0.05;
for t = 1:maxPasses*n
    if t/n>pcent
        fprintf('percent complete: %2.f%%\n',pcent*100)
        pcent = pcent+0.05;
    end
    % Choose variable to update
    i = randi(n);
    
    % Evaluate the gradient for example i
    [f,g] = logisticL2_loss(w,X(i,:),y(i),lambda);
    
    % Choose the step-size
    alpha = 1/(lambda*t);
    garray = [garray g];
    for i=1:d
        if g(i) ~= 0
            Dv(i) = 1/(sqrt(delta+sum(garray(i,:).^2)));
        end
    end
    D = spdiags(Dv',0,d,d);
    % Take the stochastic gradient step
    %grecord = [grecord; g];
    w = w - alpha*D*g;
    
    if mod(t,n) == 0
        change = norm(w-w_old,inf);
        fprintf('Passes = %d, function = %.4e, change = %.4f\n',t/n,logisticL2_loss(w,X,y,lambdaFull),change);
        if change < progTol
            fprintf('Parameters changed by less than progTol on pass\n');
            break;
        end
        break
        w_old = w;
    end
end