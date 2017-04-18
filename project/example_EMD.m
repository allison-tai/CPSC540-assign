load MNIST.mat
% earthmover distance for computer vision example

X = images;

% thing to convolve
filter = [0 -1 0; -1 0 1; 0 1 0];
filter = [1 0; 0 0];

% get random test data
Itest = randi(size(X,3));
Xtest = X(:,:,Itest); labeltest = labels(Itest);

% get random sample data for each number
samples = 3;
Isamples = []; %indices
for i = 0:9
    I = find(labels==i);
    Isamples = [Isamples; I(randsample(length(I),samples))];
end
X = X(:,:,Isamples); samplabel = Isamples; % random

% convolve
%for i=1:samples
%    Xc(:,:,i) = convolve(X(:,:,i),filter,1);
%end
%Xctest = convolve(Xtest(:,:),filter,1);
Xctest = Xtest;
Xc = X;

[n m] = size(Xctest);
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
x = reshape(Xctest,[n*m 1]); 
y = squeeze(reshape(Xc,[n*m 1 samples*10]));
tol = 0.5; lambda = 1;

[C gamma] = OTsolve(cij,x,y,tol,lambda);
% find optimal match
[Cmin xi] = min(C); 

% individual case
fprintf('Error (transportation cost) is %.3f\n',Cmin)
figure(1)
subplot(2,1,1);
imshow(X(:,:,xi)) % show match
subplot(2,1,2);
imshow(Xtest) % show test

function cij = cost(dist)
    cij = dist;
end