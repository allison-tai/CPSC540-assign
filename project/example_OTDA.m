load MNIST.mat

concavses = [];
concavwes = [];
convexses = [];
convexwes = [];
N = 30;
var = [];
for i = 1:N
    i
X = images;
n = 100; % smaller array for computation speed
ntest = 100; 

I = randsample(size(X,3),ntest*10);
Xtest = X(:,:,I); testlabels = labels(I);
I = randsample(size(X,3),n*10);
X = X(:,:,I); Xlabels = labels(I);
%X = X(:,:,I(ntest+1:n+ntest));
%[Xtest testlabels] = sampleMNIST(ntest,images,labels);
%[X Xlabels] = sampleMNIST(n,images,labels);
Itest = []; I = [];
for i=0:9
    I = [I sum(find(Xlabels==i))];
    Itest = [Itest sum(find(testlabels==i))];
end
var = [var norm(I-Itest,1)];

cost = zeros(n,n);
for i = 1:ntest*10
    for j = 1:n*10
        % rate determining step... faster way to do this calc?
        %cost(i,j) = sum(sum(X(:,:,i)~=X(:,:,j))); %
        cost2(i,j) = norm(Xtest(:,:,i)-X(:,:,j),'fro');
        cost1(i,j) = log(1+cost2(i,j));
    end
end

x = ones(ntest*10,1); y = ones(n*10,1);
tol = 0.005; lambda = 3;
[C gamma] = OTsolve(cost1,x,y,tol,lambda);

gamma = gamma*n*10; % rescale
[conf yi] = max(gamma');
concavses = [concavses sum(testlabels(yi)==Xlabels)/(ntest*10)*100];
concavwes = [concavwes sum((testlabels(yi)==Xlabels).*conf')/(sum(conf))*100];
%fprintf('For Concave cost:\n\t Success rate:\t\t %2.f %% \n',sum(testlabels(yi)==Xlabels)/(ntest*10)*100)
%fprintf('\t Weighted success rate:\t %2.f %% \n',sum((testlabels(yi)==Xlabels).*conf')/(sum(conf))*100)


[C gamma] = OTsolve(cost2,x,y,tol,lambda);

gamma = gamma*n*10; % rescale
[conf yi] = max(gamma');
convexses = [convexses sum(testlabels(yi)==Xlabels)/(ntest*10)*100];
convexwes = [convexwes sum((testlabels(yi)==Xlabels).*conf')/(sum(conf))*100];
%fprintf('For Convex cost:\n\t Success rate:\t\t %2.f %% \n',sum(testlabels(yi)==Xlabels)/(ntest*10)*100)
%fprintf('\t Weighted success rate:\t %2.f %% \n',sum((testlabels(yi)==Xlabels).*conf')/(sum(conf))*100)
end
fprintf('\t\tFor Convex Cost\t\tFor Concave Cost\nSuccess rate:\t%2.2f+/-%1.2f\t\t\t%2.2f+/-%1.2f\n'...
    ,mean(convexses),std(convexses)/sqrt(N),mean(concavses),std(concavses)/sqrt(N))
A = corrcoef(convexses,var); B = corrcoef(concavses,var);
fprintf('Correlation between discrepancy and success\nConvex: \t%2.2f\t\tConcave: \t%2.2f\n'...
    ,A(1,2),B(1,2))
return

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
       