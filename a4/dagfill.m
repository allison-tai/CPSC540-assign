clear all
load MNIST_images.mat
m = size(X,1);
n = size(X,3);

% Train an inhomogeneous markov chain
m_i = zeros(1,m); % starting distribution
p_ij = zeros(m-2,m-2,2^8);
models = cell(m,m,4);
X = sparse(X);
for j = 1:m-2
    j
    for i = 1:m-2
        X0 = X(i:i+2,j:j+2,:);
        X0 = reshape(X0(:,:,:),9,n)';
        Xij = X0(:,1:8);
        yij = X0(:,9);
        modelij = binaryTabular(Xij,yij,1);
        models(i+2,j+2,:) = struct2cell(modelij);
    end
end

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
                if j < 3
                    I(i,j) = 0;
                else
                    Zhat = I(i-2:i,j-2:j);
                    Zhat = reshape(Zhat(:,:),9,1)';
                    Xhat = Zhat(1:8);
                    model = cell2struct(models(i,j,:),...
                        {'rows','counts','sample','alpha'},3);
                    I(i,j) = model.sample(model,Xhat);
                end
            end
        end
    end
    imagesc(reshape(I,[28 28]));
end
colormap gray
