load MNIST_images.mat

% earthmover distance for computer vision example

Xtest = X(:,:,randi(size(X,3)));
X = X(:,:,(0:8)*7000+1); % distinct numbers
n = size(X,2);

% build cost matrix
cost = zeros(n^2,n^2);
for i=1:n^2
    for j = 1:n^2
        % Earth mover cost on reshapen images
        cost(i,j) = sqrt((ceil(i/n)-ceil(j/n))^2+(mod(i,n)-mod(j,n))^2); 
    end
end

% vectorize inputs and solve transportation
x = reshape(Xtest,[n^2 1]); 
y = squeeze(reshape(X,[n^2 1 9]));
tol = 0.5; lambda = 5;
[C gamma] = OTsolve(cost,x,y,tol,lambda);

% find optimal match
[Cmin xi] = min(C); 
figure
imshow(X(:,:,xi)) % show match
figure
imshow(Xtest) % show test
