function [L] = gaussian(xi,d,Sigma,mu,theta)
% fits GDA to one row of data using parameters
[n, d] = size(xi);
L = (2*pi)^(-d/2) *  det(Sigma)^(-0.5) * exp(-0.5 * ...
    (xi - mu)*(Sigma)^(-1)*(xi - mu)')*theta;
end

