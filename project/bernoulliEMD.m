load MNIST.mat
% earthmover distance for computer vision example
[m n N] = size(images);

N = 50;

% build cost matrix
dist = zeros(m*n,m*n);
for i=1:m*n
    for j = 1:m*n
        % calculate distance between points, easier way to do this?
        dist(i,j) = sqrt((ceil(i/n)-ceil(j/n))^2+(mod(i,n)-mod(j,n))^2);
    end
end
correct = [];
correctstd = [];
for models = 3:2:9
correct2 = 0;
%models = 15;
[X samplabel pi] = bernoullimodel(images,labels,1000,models);
pi = reshape(pi,[1 models*10]);
X = reshape(X,[28 28 models*10]); samplabel = reshape(samplabel,[models*10 1]);
fprintf('Number of models:\t %2d \n',models)
for i=1:N

% get random test data
Itest = randi(size(images,3));
Xtest = images(:,:,Itest); labeltest = labels(Itest);

% vectorize inputs and solve transportation
x = reshape(Xtest,[n*m 1]); 
y = squeeze(reshape(X,[n*m 1 models*10]));
tol = 0.1; lambda = 1;

[C gamma] = OTsolve(dist,x,y,tol,lambda);
% find optimal match
[Cmin xi] = min(C.*pi); 
if samplabel(xi)==labeltest
    correct2 = correct2+1;
end
% weighted average
Cost = [];
for i=0:9
    I = find(samplabel==i);
    Cost = [Cost norm(C(I).*pi(I),1)]; % sum of the cost of transporting to each of the samples
end
[Cmin index] = min(Cost);

end
correct = [correct correct2/N];

fprintf('EMD accuracy:\t %2.f %%\n',correct2/N*100)
pause(2)
end



return
% individual case
% plot
fprintf('Error (transportation cost) is %.3f\nMin cost correct? %d\n',Cmin,labeltest==samplabel(xi))
fprintf('Weighted average correct? %d\n',labeltest==index-1)
figure(1)
subplot(2,1,1);
imshow(X(:,:,xi)) % show match
subplot(2,1,2);
imshow(Xtest) % show test
pause
close