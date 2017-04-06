function [yOdds yCond N] = sampleAncestral(p0,pT)
% Monte-Carlo marginals, conditionals

nNodes = size(pT,3)+1;
n=10000; % number of samples
%y = rand(n,1)>0.5;
y = (rand(n,1)>p0(1));

for i=1:nNodes-1
    y = [y rand(n,1)>pT(y(:,i)+1,1,i)];
end
y = y;
yOdds = mean(y)';
% Part for 1.2.5
casei = y;
casei(y(:,end)~=0,:) = []; % consider case where Xd=1
yCond = mean(casei)';
N = sum(casei(:,end)==0);
end