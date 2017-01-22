function [w,f] = proxGradL1(funObj,w,lambda,maxIter)
% Minimize funOb(w) + lambda*sum(abs(w))

% Evaluate initial objective and gradient of smooth part
[f,g] = funObj(w);
funEvals = 1;

L = 1;
while funEvals < maxIter
    
    % proximal-gradient step
    alpha = 1/L;
    w_new = softThreshold(w - alpha*g,alpha*lambda);
    [f_new,g_new] = funObj(w_new);  
    funEvals = funEvals + 1;
    
    % adaptive step-size
    while f_new > f + g'*(w_new - w) + (L/2)*norm(w_new-w)^2
        L = L*2;
        alpha = 1/L;
        w_new = softThreshold(w - alpha*g,alpha*lambda);
        [f_new,g_new] = funObj(w_new);
        funEvals = funEvals + 1;
    end
    
    w = w_new;
    f = f_new;
    g = g_new;

    % Print out how we are doing
    optCond = norm(w-softThreshold(w-g,lambda),'inf');
    fprintf('%6d %15.5e %15.5e %15.5e\n',funEvals,alpha,f + lambda*sum(abs(w)),optCond);
    
    if optCond < 1e-1
        break;
    end
end
end

function [w] = softThreshold(w,threshold)
    w = sign(w).*max(0,abs(w)-threshold);
end
