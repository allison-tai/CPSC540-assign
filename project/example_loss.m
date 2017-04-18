load MNIST.mat

X = images;

ntest = 20;
% get random test data
Itest = randi(size(X,3),[1 ntest]);
Xtest = X(:,:,Itest); testlabel = labels(Itest)';

% get random sample data for each number
samples = 2;
Isamples = []; %indices
for i = 0:9
    I = find(labels==i);
    Isamples = [Isamples I(randsample(length(I),samples))];
end
X = X(:,:,Isamples); samplabel = labels(Isamples); % random

[n m ntest] = size(Xtest);
% build cost matrix
dist = zeros(n*m,n*m);
cij = zeros(n*m,n*m);
for i=1:n*m
    for j = 1:n*m
        % calculate distance between points, easier way to do this?
        dist(i,j) = sqrt((ceil(i/n)-ceil(j/n))^2+(mod(i,n)-mod(j,n))^2);
    end
end
cij = cost(dist);

% vectorize inputs and solve transportation
x = squeeze(reshape(Xtest,[n*m ntest])); 
y = squeeze(reshape(X,[n*m 1 samples*10]));
tol = 0.5; lambda = 2;

C = []; gamma=zeros(n*m,n*m,samples*10,ntest);
tic
for i=1:ntest
    [Ci gammai] = OTsolve(cij,x(:,i),y,tol,lambda);
    C = [C Ci'];
    %gamma(:,:,:,i)=gammai;
end
toc

% find optimal match
[Cmin xi] = min(C); 

[Loss error] = lossEMD(xi,samplabel,testlabel);
fprintf('Percent mistakes: %2f',Loss/ntest*100)
for i = 1:length(error)
    %fprintf('Error (transportation cost) is %.3f\n',Cmin())
    figure(1)
    subplot(2,1,1);
    imshow(X(:,:,xi(error(i)))) % show match
    subplot(2,1,2);
    imshow(Xtest(:,:,error(i))) % show test
    pause(2)
end


function cij = cost(dist)
    cij = dist;
end