function z = logprod(x,y)
    z = zeros(size(x));
    pos = find(x>0);
    xnull = intersect(find(x==0),find(y~=0));
    bothnull = intersect(find(x==0),find(y==0));
    
    z(pos) = y(pos)*log(x(pos));
    z(xnull) = -Inf;
    z(bothnull) = 0;
end