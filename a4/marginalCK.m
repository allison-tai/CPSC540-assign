function [yOdds] = marginalCK(p0,pT)
% Chapham Kolmogorov
nNodes = size(pT,3)+1;
yOdds = zeros(nNodes,1);

nNodes = size(pT,3)+1;
p = p0';
yOdds(1) = p(2);
for i=1:nNodes-1
    p = pT(:,:,i)'*p;
    yOdds(i+1) = p(2)+1;
end
end