function [C gamma] = OTsolve(cost,x,y,tol,lambda)
    %OTSOLVE Solves for the regularized optimal transport between vectors x and
    %multiple vectors contained in y.
    
    n = length(x);
    % normalize
    x = x/sum(x);
    stuff = ones(size(y,1),1)*sum(y); 
    y = y./stuff;

    % apply sk from negentropy paper
    I = (x>0);
    % I assume x tells me which pixels are lit up
    % I only look at the transportation of white pixels, hence cost(I,:)
    % I need to know the optimal destination of each pixel
    % Or just which direction each pixel moved
    x = x(I); cost = cost(I,:); K = exp(-lambda*cost);
    u = ones(length(x),size(y,2))/length(x);
    Ktilde = bsxfun(@rdivide,cost,x);
    uold = Inf;
    while norm(u-uold)>tol
        uold = u;
        u = 1./(Ktilde*(y./(K'*u)));
    end
    v = y./(K'*u);
    % without the sum, I get the cost of transporting each white pixel of
    % x, but no idea where the destination of that pixel is
    C = sum(u.*((K.*cost)*v));

    % construct gamma (diffusion)
    gamma = zeros(n,size(y,1),size(y,2));
    for i = 1:size(y,2)
        du = diag(u(:,i));
        dv = diag(v(:,i));
        gamma(find(I),:,i) = du*K*dv; % includes empty indices
    end
end

