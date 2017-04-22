clear all
load MNIST.mat

concavses = [];
concavwes = [];
convexses = [];
convexwes = [];
lambdastore = [];
convexmean = []; concavmean = [];
convexerror = []; concaverror = [];
N = 30;
var = [];
for index = -5:4
for i = 1:N
X = images;
n = 0.3; % number of random samples/10
ntest = 3.3; % number of total samples

% random samples
I = randsample(size(X,3),n*10);
Xtest = X(:,:,I); testlabels = labels(I);
I = randsample(size(X,3),n*10);
X = X(:,:,I); Xlabels = labels(I);
I = 0; Itest = 0;

% hybrid dataset
[Y Ylabels] = sampleMNIST(ntest-n,images,labels);
X = cat(3,X,Y); Xlabels = [Xlabels; Ylabels];
[Y Ylabels] = sampleMNIST(ntest-n,images,labels);
Xtest = cat(3,Xtest,Y); testlabels = [testlabels; Ylabels];
n=ntest;

Itest = []; I = [];
for i=0:9
    I = [I sum(find(Xlabels==i))];
    Itest = [Itest sum(find(testlabels==i))];
end
var = [var norm(I-Itest,1)];

cost = [];
for i = 1:ntest*10
    for j = 1:n*10
        % rate determining step... faster way to do this calc?
        %cost(i,j) = sum(sum(X(:,:,i)~=X(:,:,j))); %
        cost2(i,j) = norm(Xtest(:,:,i)-X(:,:,j),'fro')^2;
        %cost1(i,j) = log(1+cost2(i,j));
        cost1(i,j) = atan(cost2(i,j));
    end
end

x = ones(ntest*10,1); y = ones(n*10,1);
tol = 0.005; lambda = 1.3^index;
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
lambdastore = [lambdastore lambda];
convexmean = [convexmean mean(convexses)]; concavmean = [concavmean mean(concavses)];
convexerror = [convexerror std(convexses)/sqrt(N)];
concaverror = [concaverror std(concavses)/sqrt(N)];
convexses = []; concavses = [];
end

figure; hold on; x = lambdastore; y = convexmean; err = convexerror;
set(gca,'FontSize', 20);
title('Concave vs. Convex cost against \lambda');
xlabel('\lambda'); ylabel('% Accuracy');
axis([-Inf Inf 0 100])
patch([x fliplr(x)],[y+err fliplr(y-err)],[0.9 0.7 0.7],'EdgeColor','none','FaceAlpha',.5);
convex = plot(x,y,'r.-');
y = concavmean; err = concaverror;
patch([x fliplr(x)],[y+err fliplr(y-err)],[0.7 0.7 0.9],'EdgeColor','none','FaceAlpha',.5);
concave = plot(x,y,'b.-');
legend([convex concave],'Convex Cost','Concave Cost')