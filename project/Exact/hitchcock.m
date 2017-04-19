function x = hitchcock(wi,wf,c)
    n = length(wi); m = length(wf);
    I=eye(n);
    M = []; a = [];
    for i=1:m
        Ji = zeros(m,n);
        Ji(i,:) = ones(1,n);
        Mi = [I; -I; Ji; -Ji];
        M = [M Mi];
        for j=1:n
            c(j,i)
            j
            a(n*(i-1)+j)=c(j,i);
        end
    end
    b = [wi; -wi; wf; -wf];
    y0 = zeros(n*m,1);
    options = optimoptions(@fmincon,'Algorithm','sqp','MaxIterations',1500,'Display','off');
    y = fmincon(@(y) a*y,y0,-M,-b,[],[],y0,ones(n*m,1)*max([max(wi) max(wf)]),[],options);
    %y = fmincon(@(y) a*y,y1,-M,-b,[],[],y0,ones(n*m,1)*max([max(wi) max(wf)]),[],options);
    x=rand(n,m);
    for i=1:m
        for j=1:n
            x(j,i)=y(n*(i-1)+j);
        end
    end
end