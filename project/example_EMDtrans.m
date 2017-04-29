load transMNIST.mat
% earthmover distance for computer vision example

X = transImages;

% get random test data
Itest = randi(size(X,3));
Xtest = X(:,:,Itest); labeltest = labels(Itest);

% get random sample data for each number
samples = 3; % 3 samples of each number
index = 0:9;
load MNIST
[X samplabel pi] = bernoullimodel(images,labels,1000,samples); % obtain samples 

[n m N] = size(X);

% build cost matrix
dist = zeros(n*m,n*m);
cij = zeros(n*m,n*m);
for i=1:n*m
    for j = 1:n*m
        % calculate distance between points, easier way to do this?
        dist(i,j) = sqrt((ceil(i/n)-ceil(j/n))^2+(mod(i,n)-mod(j,n))^2);
    end
end
cij = dist;

% vectorize inputs and solve transportation
x = reshape(Xtest,[n*m 1]);
y = squeeze(reshape(X,[n*m 1 samples*10]));
tol = 0.001; lambda = 1;

% do OT for translation checks
[C gamma] = OTsolve(cij,x,y,tol,lambda);
multX = pixelTrans(gamma, x, n);

% now do OT again, for real (doing y one at a time - sort of inefficient)
for j = 1:samples*10
   [Cj gamma] = OTsolve(cij,multX(:,j),y(:,j),tol,lambda);
   C(:,j) = Cj;
end

% find optimal match
P = 1./C;
[Cmin xi] = min(C); 
 
% individual case
fprintf('Error (transportation cost) is %.3f\n',Cmin)
figure(1)
subplot(2,1,1);
imshow(X(:,:,xi)) % show match
subplot(2,1,2);
imshow(Xtest) % show test
pause(1)
close
return
for i=1:30
    figure(1)
    imshow(reshape(multX(:,i),[28 28]))
    pause
end