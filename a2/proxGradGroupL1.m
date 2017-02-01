function [w,f] = proxGradGroupL1(funObj,w,lambda,maxIter)
% Minimize funOb(w) + lambda*sum(abs(w)) in groups % rows 0

% Evaluate initial objective and gradient of smooth part
[f,g] = funObj(w);
funEvals = 1;

L = 1;
while funEvals < maxIter
    
    % proximal-gradient step
    alpha = 1/L;
    w_new = softThresholdg(w - alpha*g,alpha*lambda); % change this function for group
    [f_new,g_new] = funObj(w_new);  
    funEvals = funEvals + 1;
    
    % adaptive step-size
    while f_new > f + g'*(w_new - w) + (L/2)*norm(w_new-w)^2
        L = L*2;
        alpha = 1/L;
        w_new = softThresholdg(w - alpha*g,alpha*lambda);
        [f_new,g_new] = funObj(w_new);
        funEvals = funEvals + 1;
    end
    
    w = w_new;
    f = f_new;
    g = g_new;

    % Print out how we are doing
    optCond = norm(w-softThresholdg(w-g,lambda),'inf');
    fprintf('%6d %15.5e %15.5e %15.5e\n',funEvals,alpha,f + lambda*sum(abs(w)),optCond);
    
    if optCond < 1e-1
        break;
    end
end
end

function [w] = softThresholdg(w,threshold) % change to group norm
    [d,k] = size(w);
    for i=1:d
        groupnorm_i = sqrt(sum(w(i,:).^2));
        if groupnorm_i ~= 0
            groupnorm(i) = groupnorm_i;
        else
            groupnorm(i) = 1;
        end
    end
    groupnorm = groupnorm'*ones(1,k); 
    w = (w./groupnorm).*max(0,groupnorm-threshold);
end
