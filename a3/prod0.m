function [prod] = prod0(x,y)
% Returns x.*y, but sets the result to 0 at indices where x==0.
prod = x.*y;
prod(x(:)==0) = 0;