function [yOdds] = sampleBackwards(pd,pT)

% Set up our DP table
d = size(pT,3)+1;
% number of samples
n = 10000;

% fill in DP table
y = zeros(n,d);
y(:,d) = rand(n,1) > pd(1)*ones(n,1);
% sample from our distribution, backwards (we know there is 0 chance of
% being in state 2)
for j=d-1:-1:1
<<<<<<< HEAD
    yp = pT(:,y(:,j+1)+1,j)';
=======
    M(j,:) = M(j+1,:)*pT(:,:,j)';
>>>>>>> 24a1d3b33f570107428384b8f98bec9855924f9f
    % normalize
    yp = yp./(sum(yp,2)*ones(1,2));
    %samples
    y(:,j) = rand(n,1) > yp(:,1);
end
yOdds = mean(y)';

% sample from our distribution, backwards (we know there is 0 chance of
% being in state 2)
%y = zeros(n,d);
%for j=d-1:-1:1
%        y(:,j) = rand(n,1) > M(j,y(:,j+1)+1)';
%end
end