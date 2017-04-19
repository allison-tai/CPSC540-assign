clear all;

n = 3;
m = 3;

xi = [3 8; 5 3; 4 4; 4 7]/10; n = size(xi,1);
%yj = [1.2 1.2; 2.3 3.1; 4.2 4.8; 2.8 1.9; 0.5 2]/10; m =size(yj,1);
yj = xi(1:n,:)-(rand(n,2)+1)/8; %yj = [yj; xi(1,:)-rand(1,2)/10];
m =size(yj,1);
%wi = ones(1,n)'; wj = ones(1,m)'*n/m;
wi = rand(n,1)+1; wj = wi+rand(1,n); wj = wj/sum(wj);
xi = [xi; 0.9 0.9]; n = size(xi,1);
wi = [wi; 0.2]
wi = wi/sum(wi); 
%xi = rand(n,2); 
%yj = rand(m,2); 

pij = [];
for i=1:n
    for j=1:m
        cij(i,j) = log(1+norm(xi(i,:)-yj(j,:)));
        cij2(i,j) = norm(xi(i,:)-yj(j,:))^2;
        pij(i,j,:,:) = (0:0.1:1)'*xi(i,:)+(1-(0:0.1:1))'*yj(j,:);
    end
end

gamma = hitchcock(wi,wj,cij);
gamma(gamma<0.001)=0
figure(1)
set(gca,'FontSize', 20);
axis([0 1 0 1]); hold on;
scatter(xi(:,1),xi(:,2),wi*600,'g','filled'); hold on;
scatter(yj(:,1),yj(:,2),wj*600,'r','filled');hold on;
for i=1:n
    for j=1:m
        if gamma(i,j)>0.01
            plot(squeeze(pij(i,j,:,1)),squeeze(pij(i,j,:,2)),'k','LineWidth',10*gamma(i,j)); hold on;
        end
    end
end
title('Concave Cost Function with Perturbation')

gamma = hitchcock(wi,wj,cij2);
gamma(gamma<0.004)=0
figure(2)
set(gca,'FontSize', 20);
axis([0 1 0 1]); hold on;
scatter(xi(:,1),xi(:,2),wi*600,'g','filled'); hold on;
scatter(yj(:,1),yj(:,2),wj*600,'r','filled');hold on;
for i=1:n
    for j=1:m
        if gamma(i,j)>0
            plot(squeeze(pij(i,j,:,1)),squeeze(pij(i,j,:,2)),'k','LineWidth',10*gamma(i,j)); hold on;
        end
    end
end
title('Convex Cost Function with Perturbation')