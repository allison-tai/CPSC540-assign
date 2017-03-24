function [y_best, M, T] = viterbiDecode(p0,pT)

d = size(pT,3)+1;
M = zeros(2,d);
T = zeros(2,d);

M(:,1) = p0;

% memoize
for i=1:d-1
    for s=1:2
        m = [M(1,i)*pT(1,s,i),M(2,i)*pT(2,s,i)];
        [M(s,i+1), T(s,i+1)] = max(m);
    end
end

y_best = zeros(d,1);
[mx, argmx] = max(M(:,d));
y_best(d,1) = argmx;
for i=d-1:-1:1
    y_best(i,1) = T(y_best(i+1,1), i+1);
end
end