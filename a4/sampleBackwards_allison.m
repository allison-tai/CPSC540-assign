function [yOdds yCond] = sampleBackwards(p0,pT)

% Set up our DP table
d = size(pT,3)+1;
M = ones(d,2);
% number of samples
n = 10000;

% fill in DP table
M(d,:) = [1 0];
% sample from our distribution, backwards (we know there is 0 chance of
% being in state 2)
y = zeros(n,d);
for j=d-1:-1:1
    M(j,:) = M(j+1,:)*pT(:,:,j);
    % normalize
    M(j,:) = M(j,:)/sum(M(j,:),2);
    % sample
    y(:,j) = rand(n,1) > M(j,y(:,j+1)+1)';
end
yOdds = mean(y)';

% sample from our distribution, backwards (we know there is 0 chance of
% being in state 2)
%y = zeros(n,d);
%for j=d-1:-1:1
%        y(:,j) = rand(n,1) > M(j,y(:,j+1)+1)';
%end
end