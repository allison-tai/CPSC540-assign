load MNIST_images.mat

n = 100; % smaller array for computation speed
X = X(:,:,randsample(size(X,3),n)); % random

cost = zeros(n,n);
for i = 1:n
    for j = 1:n
        %cost(i,j) = sum(sum(X(:,:,i)~=X(:,:,j))); %
        cost(i,j) = norm(X(:,:,1)-X(:,:,2),'fro'); %
    end
end

% need to minimize gamma.*cost+entropy(gamma);
       