load MNIST_images.mat

n = 30; % smaller array for computation speed
ntest = 10; 

I = randsample(size(X,3),n+ntest)
Xtest = X(:,:,I(1:ntest));
X = X(:,:,I(ntest+1:n+ntest));


cost = zeros(n,n);
for i = 1:ntest
    for j = 1:n
        % rate determining step... faster way to do this calc?
        %cost(i,j) = sum(sum(X(:,:,i)~=X(:,:,j))); %
        cost(i,j) = norm(Xtest(:,:,i)-X(:,:,j),'fro'); 
    end
end

x = ones(ntest,1);
y = ones(n,1);
tol = 0.5; lambda = 5;
[C gamma] = OTsolve(cost,x,y,tol,lambda);

[conf yi] = max(gamma')
for i=1:ntest
    if conf(i)>0.5 % confidence level best match
        figure(1)
        fprintf('Confidence for sample %d is %f\n',i,conf(i))
        subplot(2,1,1);
        imshow(X(:,:,yi(i)))
        subplot(2,1,2); 
        imshow(Xtest(:,:,i))
        pause(3)
    end
end
       