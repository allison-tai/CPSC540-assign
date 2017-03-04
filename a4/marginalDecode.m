function [maxmarg] = marginalDecode(p0,pT)
% Chapham Kolmogorov
nNodes = size(pT,3)+1;

nNodes = size(pT,3)+1;
p = p0';
[pmax M] = max(p0);
maxmarg = [M-1];
for i=1:nNodes-1
    p = pT(:,:,i)'*p;
    [pmax M] = max(p);
    maxmarg = [maxmarg M-1]
end
end