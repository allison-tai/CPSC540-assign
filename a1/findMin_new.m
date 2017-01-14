function [w,f] = findMin_new(funObj,w,maxEvals,verbose,varargin)
% Find local minimizer of differentiable function

% Parameters of the Optimizaton
optTol = 1e-2;
gamma = 1e-4;

% Evaluate the initial function value and gradient
[f,g,h] = funObj(w,varargin{:});
funEvals = 1;
backit = 0;
alpha = 1;
%X = varargin{1}; lambda = varargin{end}; L = 1/4*max(eigs(X*X'))+lambda; alpha = 1/L; % 3.
%alphaFinal = alpha;
while 1
    %% Compute search direction
    d = h\g; % 4.
    %d = g;
    
    %% Line-search to find an acceptable value of alpha
	w_new = w - alpha*d;
	[f_new,g_new,h_new] = funObj(w_new,varargin{:});
	funEvals = funEvals+1;
    
    dirDeriv = g'*d;
    while f_new > f - gamma*alpha*dirDeriv
        backit = backit + 1;
        if verbose
            fprintf('Backtracking... %dth time\n',backit);
        end
        alpha = alpha^2*dirDeriv/(2*(f_new - f + alpha*dirDeriv)); % cubic-Hermite interpolation
        %alpha = alpha/2; % 1.
        w_new = w - alpha*d;
        [f_new,g_new,h_new] = funObj(w_new,varargin{:});
        funEvals = funEvals+1;
    end
    alphaFinal = alpha;

    %% Update step-size for next iteration
    alpha = 1; 
    %v = g_new-g; alpha = -alpha*v'*g/(v'*v); % 2.
    
    %% Sanity check on step-size
    if ~isLegal(alpha) || alpha < 1e-10 || alpha > 1e10
       alpha = 1; 
    end
    
    %% Update parameters/function/gradient
    w = w_new;
    f = f_new;
    g = g_new;
    h = h_new;
	
    %% Test termination conditions
	optCond = norm(g,'inf');
    if verbose
        fprintf('%6d %6d %15.5e %15.5e %15.5e\n',funEvals,backit,alphaFinal,f,optCond);
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