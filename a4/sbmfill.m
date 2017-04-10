clear all
load MNIST_images.mat
m = size(X,1);
n = size(X,3);
N = 4; M = N-1; % NxN DAG train

models = cell(m,m,3);
% train
for j = 1:m
    for i = 1:m
        X0 = X(1:i,1:j,:);
        X0 = reshape(X0(:,:,:),i*j,n)';
        %X0 = sparse(reshape(X0(:,:,:),i*j,n)');
        Xij = X0(:,1:i*j-1);
        yij = X0(:,i*j);
        % logistic regression needs {-1,1} encoding instead of {0,1} encoding
        % for y
        yij(yij == 0) = -1;
        modelij = logisticL2(Xij,yij,1);
        models(i,j,:) = struct2cell(modelij);
    end
    j
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
                Zhat = I(1:i,1:j);
                Zhat = reshape(Zhat(:,:),i*j,1)';
                Xhat = Zhat(1:i*j-1);
                model = cell2struct(models(i,j,:),...
                        {'w','predict','sample'},3);
                yhat = model.sample(model,Xhat);
                if yhat == -1
                    I(i,j) = 0;
                elseif yhat == 1
                    I(i,j) = 1;
                end
            end
        end
    end
    imagesc(reshape(I,[28 28]));
end
colormap gray
