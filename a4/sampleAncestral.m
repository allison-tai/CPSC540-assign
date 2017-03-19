function [yOdds yCond N] = sampleAncestral(p0,pT)
% Monte-Carlo marginals, conditionals

nNodes = size(pT,3)+1;
n=10000; % number of samples
y = (rand(n,1)>p0(1));

for i=1:nNodes-1
    y = [y rand(n,1)>pT(y(:,i)+1,1,i)];
end
y = y+1;
yOdds = mean(y)';
% Part for 1.2.5
yCond = [];
N = sum(y)'; % number of occurences of Xd=1
for i=1:nNodes
    % consider case where Xi=1
    casei = y;
    casei(y(:,i)~=1,:) = [];
    yCond = [yCond; mean(casei)];
end
end