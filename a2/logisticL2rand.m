function [model steps] = logisticL2rand(X,y,lambda)

% Add bias variable
[n,d] = size(X);
X = [ones(n,1) X];
d = d+1;

% Initial values of regression parameters
w = zeros(d,1);

% Optimizaion parameters
maxPasses = 500;
progTol = 1e-4;
%L = .25*max(eig(X'*X)) + lambda;
L = .25*diag(X'*X) + lambda;
p = L/norm(L,1); % probability distribution
steps = [];
w_old = w;
for t = 1:maxPasses*d
    
    % Choose variable to update 'j'
    %j = sampleDiscrete(p); % 2. proportionate sampling
    j = randi(d); % 3. uniform sampling
    Z = X(:,j);
    L_j = L(j);
    
    % Compute partial derivative 'g_j'
    yXw = y.*(X*w);
    sigmoid = 1./(1+exp(-yXw));
    g = -Z'*(y.*(1-sigmoid)) + lambda*w;
    g_j = g(j);
    
    % Update variable
    w(j) = w(j) - (1/L_j)*g_j;
    
    % Check for lack of progress after each "pass"
    if mod(t,d) == 0
        change = norm(w-w_old,inf);
        fprintf('Passes = %d, function = %.4e, change = %.4f, step = %.3e\n',t/d,logisticL2_loss(w,X,y,lambda),change,(1/L_j)*g_j);
        steps = [steps (1/L_j)*g_j];
        if change < progTol
            fprintf('Parameters changed by less than progTol on pass\n');
            break;
        end
        w_old = w;
    end
end


model.w = w;
model.predict = @predict;
end

function [yhat] = predict(model,Xhat)
[t,d] = size(Xhat);
Xhat = [ones(t,1) Xhat];
w = model.w;
yhat = sign(Xhat*w);
end


