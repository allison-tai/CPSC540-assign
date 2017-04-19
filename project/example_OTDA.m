load MNIST.mat

X = images;
n = 3; % smaller array for computation speed
ntest = 3; 

%Xtest = X(:,:,I(1:ntest));
%X = X(:,:,I(ntest+1:n+ntest));
[Xtest testlabels] = sampleMNIST(ntest,images,labels)
[X Xlabels] = sampleMNIST(n,images,labels)

cost = zeros(n,n);
for i = 1:ntest*10
    for j = 1:n*10
        % rate determining step... faster way to do this calc?
        %cost(i,j) = sum(sum(X(:,:,i)~=X(:,:,j))); %
        cost(i,j) = log(1+norm(Xtest(:,:,i)-X(:,:,j),'fro')); 
    end
end

x = ones(ntest*10,1);
y = ones(n*10,1);
tol = 0.005; lambda = 1;
[C gamma] = OTsolve(cost,x,y,tol,lambda);

gamma = gamma*n*10; % rescale
[conf yi] = max(gamma');
fprintf('Success rate: %2.f %% \n',sum(testlabels(yi)==Xlabels)/(ntest*10)*100)
fprintf('Weighted success rate: %2.f %% \n',sum((testlabels(yi)==Xlabels).*conf')/(sum(conf))*100)
for i=1:ntest*10
    %if conf(i)<mean(conf) % confidence level best match
        figure(1)
        fprintf('Confidence for sample %d is %f\t Is correct? %d\n',i,conf(i),(testlabels(yi(i))==Xlabels(i)))
        subplot(2,1,1);
        imshow(X(:,:,yi(i)))
        subplot(2,1,2); 
        imshow(Xtest(:,:,i))
        pause(1)
        close
    %end
end
       