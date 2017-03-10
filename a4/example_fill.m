clear all
load MNIST_images.mat
m = size(X,1);
n = size(X,3);

% Train an independent Bernoulli model
p_ij = zeros(m,m);
for i = 1:m
    for j = 1:m
        p_ij(i,j) = sum(X(i,j,:) == 1)/n;
    end
end
figure(1);
imagesc(p_ij);

% Fill-in some random test images
t = size(Xtest,3);
figure(2);
for image = 1:4
    subplot(2,2,image);
    
    % Grab a random test example
    ind = randi(t);
    I = Xtest(:,:,ind);
        
    % Fill in the bottom half using the model
    for i = 1:m
        for j = 1:m
            if isnan(I(i,j))
                I(i,j) = rand < p_ij(i,j);
            end
        end
    end
    imagesc(reshape(I,[28 28]));
end
colormap gray
