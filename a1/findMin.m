function [w,f] = findMin(funObj,w,maxEvals,verbose,varargin)
% Find local minimizer of differentiable function

% Parameters of the Optimizaton
optTol = 1e-2;
gamma = 1e-4;

% Evaluate the initial function value and gradient
[f,g] = funObj(w,varargin{:});
funEvals = 1;

alpha = 1;
while 1
    %% Compute search direction
    d = g;
    
    %% Line-search to find an acceptable value of alpha
	w_new = w - alpha*d;
	[f_new,g_new] = funObj(w_new,varargin{:});
	funEvals = funEvals+1;
    
    dirDeriv = g'*d;
    while f_new > f - gamma*alpha*dirDeriv
        if verbose
            fprintf('Backtracking...\n');
        end
        alpha = alpha^2*dirDeriv/(2*(f_new - f + alpha*dirDeriv));
        w_new = w - alpha*d;
        [f_new,g_new] = funObj(w_new,varargin{:});
        funEvals = funEvals+1;
    end
    alphaFinal = alpha;

    %% Update step-size for next iteration
    alpha = 1;
    
    %% Sanity check on step-size
    if ~isLegal(alpha) || alpha < 1e-10 || alpha > 1e10
       alpha = 1; 
    end
    
    %% Update parameters/function/gradient
    w = w_new;
    f = f_new;
    g = g_new;
	
    %% Test termination conditions
	optCond = norm(g,'inf');
    if verbose
        fprintf('%6d %15.5e %15.5e %15.5e\n',funEvals,alphaFinal,f,optCond);
    end
	
	if optCond < optTol
        if verbose
            fprintf('Problem solved up to optimality tolerance\n');
        end
		break;
	end
	
	if funEvals >= maxEvals
        if verbose
            fprintf('At maximum number of function evaluations\n');
        end
		break;
	end
end
end

function [legal] = isLegal(v)
legal = sum(any(imag(v(:))))==0 & sum(isnan(v(:)))==0 & sum(isinf(v(:)))==0;
end