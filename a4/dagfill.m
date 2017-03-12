clear all
load MNIST_images.mat
m = size(X,1);
n = size(X,3);
N = 3; M = N-1; % NxN DAG train

% Train an inhomogeneous markov chain on first three columns
m_i = zeros(1,m); % starting distribution
p_ij = zeros(m-2,m-2,2);
for j = 1:N
    m_i(j) = sum(X(1,j,:))/n;
    for i = 1:m-1
        X0 = squeeze(X(i:i+1,j,:));
        X1 = X0;
        X0(:,X0(1,1,:)~=0) = [];
        X1(:,X1(1,1,:)~=1) = [];
        n1 = sum(X1(1,:));
        p_ij(i,j,1) = sum(X0(2,:) == 1)/max(n-n1,1);
        p_ij(i,j,2) = sum(X1(2,:) == 1)/max(n1,1);
    end
end
% train NxN DAG on remainder
models = cell(m,m,4);
for j = 1:m-M
    for i = 1:m-M
        X0 = X(i:i+M,j:j+M,:);
        X0 = reshape(X0(:,:,:),N^2,n)';
        Xij = X0(:,1:N^2-1);
        yij = X0(:,N^2);
        modelij = binaryTabular(Xij,yij,1);
        models(i+M,j+M,:) = struct2cell(modelij);
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
                if j < N
                    I(i,j) = rand < p_ij(i-1,j,I(i-1,j)+1);
                else
                    Zhat = I(i-M:i,j-M:j);
                    Zhat = reshape(Zhat(:,:),N^2,1)';
                    Xhat = Zhat(1:N^2-1);
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
