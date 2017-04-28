load transMNIST.mat
% earthmover distance for computer vision example

X = transImages;

% thing to convolve
filter = [1];
%filter = [1 0; 0 0];

% get random test data
Itest = randi(size(X,3));
Xtest = X(:,:,Itest); labeltest = labels(Itest);

% get random sample data for each number
samples = 3; % 3 samples of each number
index = 0:9;
[X samplabel] = sampleMNIST(n,transImages,labels, index);


% convolve
Xc = [];
for i=1:samples*10
    Xc(:,:,i) = convolve(X(:,:,i),filter,1);
end
Xctest = convolve(Xtest(:,:),filter,1);

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
cij = dist;

% vectorize inputs and solve transportation
x = reshape(Xctest,[n*m 1]);
y = squeeze(reshape(Xc,[n*m 1 samples*10]));
tol = 0.1; lambda = 1.5;

% run OT and solve for best
[C gamma] = OTsolve(cij,x,y,tol,lambda);

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

%function cij = cost(dist)
 %   cij = dist;
%end