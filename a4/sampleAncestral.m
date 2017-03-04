function [yOdds] = sampleAncestral(p0,pT)
% Exact decoding of (short) Markov binary chains

nNodes = size(pT,3)+1;
n=1000;
% Initial value of states to try
yOdds = zeros(nNodes,1);
y = (rand(n,1)>p0(1));

yOdds(1)=mean(y);

maxProb = -inf;
for i=1:nNodes-1
    % Evaluate this sequence of states
    y = rand(n,1)>pT(y+1,1,i);
    yOdds(i+1) = mean(y);
end
end