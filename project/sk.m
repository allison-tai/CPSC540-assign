% works only with square nonnegative matrix A, with not too many zeros
function [P, c, r] = sk(A, iter, tol)
    k = 1;
    c = 1./sum(A);
    r = 1./(A*c');
    while k < iter
        k = k + 1;
        cinv = (A'*r);
        % if the algorithm has converged to a certain point, stop
        if max(abs(cinv.*c - 1)) <= tol
            break
        end
        c= 1./cinv
        r = 1./(A*c);
    end
    P = A.*(r*c); 
end