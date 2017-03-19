function [maxmarg] = marginalDecode(p0,pT)
nNodes = size(pT,3)+1;

p = p0';
[pmax M] = max(p0);
maxmarg = [M]; % because matlab doesn't count 0
for i=1:nNodes-1
    p = pT(:,:,i)'*p;
    [pmax M] = max(p);
    maxmarg = [maxmarg; M];
end
end