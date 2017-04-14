load MNIST_images.mat

n = 400; % smaller array for computation speed
X = X(:,:,randsample(size(X,3),n)); % random

cost = zeros(n,n);
for i = 1:n
    for j = 1:n
        cost(i,j) = norm(X(:,:,i)-X(:,:,j));
    end
end

