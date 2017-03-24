function [p] = forwardBackwards(p0,pT)

% Set up our DP table
d = size(pT,3)+1;
V = forwards(p0,pT,d);
B = backwards(pT, d);
p = zeros(d,2);

for j=1:d
   p(j,:) = V(j,:).*B(j,:);
   % Normalize
   p(j,:) = p(j,:)/sum(p(j,:),2);
end
end

function [M] = forwards(p0,pT,d)
M = zeros(d,2);
M(1,:) = p0';
for j=2:d
    M(j,1) = M(j-1,1)*pT(1,1,j-1) + M(j-1,2)*pT(2,1,j-1);
    M(j,2) = M(j-1,1)*pT(1,2,j-1) + M(j-1,2)*pT(2,2,j-1);
end 
end

function [M] = backwards(pT,d)
M = ones(d,2);
M(d,:) = [1 0];
for j=d-1:-1:1
    M(j,1) = M(j+1,1)*pT(1,1,j) + M(j+1,2)*pT(1,2,j);
    M(j,2) = M(j+1,1)*pT(1,2,j) + M(j+1,2)*pT(2,2,j);
end
end