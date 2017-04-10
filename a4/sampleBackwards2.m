function [yOdds] = sampleBackwards2(pd,pT)

% Set up our DP table
d = size(pT,3)+1;
V = zeros(d,2);

% number of samples
n = 10000;

% fill in DP table
V(end,:) = pd;
y = zeros(n,d);
y(:,d) = rand(n,1) > V(d,1);
% sample from our distribution, backwards (we know there is 0 chance of
% being in state 2)
for j=d-1:-1:1
    V(j,:) = pT(:,:,j)*V(j+1,:)';
    % normalize
    V(j,:) = V(j,:)/sum(V(j,:));
    %samples
    y(:,j) = rand(n,1) > V(j,1);
end
yOdds = mean(y)';

% sample from our distribution, backwards (we know there is 0 chance of
% being in state 2)
%y = zeros(n,d);
%for j=d-1:-1:1
%        y(:,j) = rand(n,1) > M(j,y(:,j+1)+1)';
%end
end