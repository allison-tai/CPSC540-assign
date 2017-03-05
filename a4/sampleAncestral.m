function [yOdds yCond N] = sampleAncestral(p0,pT)
% Monte-Carlo marginals, conditionals

nNodes = size(pT,3)+1;
n=1000;
% Initial value of states to try
yOdds = zeros(nNodes,1);
% yOdds = zeros(nNodes,nNodes);
y = (rand(n,1)>p0(1));
yOdds(1)=mean(y);

for i=1:nNodes-1
    y = [y rand(n,1)>pT(y(:,i)+1,1,i)];
    yOdds(i+1) = mean(y(:,i+1));
end
% Part for 1.2.5
yCond = [];
N = sum(y)'; % number of occurences of Xi=1
for i=1:nNodes
    % consider case where Xi=1
    casei = y;
    casei(y(:,i)~=1,:) = [];
    yCond = [yCond; mean(casei)];
end