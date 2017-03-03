function [prob] = decodeProb(y,p0,pT)
nNodes = size(pT,3)+1;
prob = p0(y(1));
for t = 2:nNodes
    prob = prob*pT(y(t-1),y(t),t-1);
end
end