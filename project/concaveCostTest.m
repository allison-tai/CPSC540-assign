clear all
load MNIST.mat

concavses = []; convexses = []; distses = [];
lambdastore = [];
convexmean = []; concavmean = []; distmean = [];
convexerror = []; concaverror = []; disterror = [];
N = 30;
var = [];
for index = -5:6
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
[Y Ylabels] = sampleMNIST(ntest-n,images,labels,0:9);
X = cat(3,X,Y); Xlabels = [Xlabels; Ylabels];
[Y Ylabels] = sampleMNIST(ntest-n,images,labels,0:9);
Xtest = cat(3,Xtest,Y); testlabels = [testlabels; Ylabels];
n=ntest;

Itest = []; I = [];
for i=0:9
    I = [I sum(find(Xlabels==i))];
    Itest = [Itest sum(find(testlabels==i))];
end
var = [var norm(I-Itest,1)];

dist = [];
for i = 1:ntest*10
    for j = 1:n*10
        dist(i,j) = norm(Xtest(:,:,i)-X(:,:,j),'fro'); % calculate distances
    end
end
cost1 = log(1+dist);
cost2 = dist.^2;

x = ones(ntest*10,1); y = ones(n*10,1);
tol = 0.005; lambda = 1.3^index;

% solve OT
[C gamma] = OTsolve(dist,x,y,tol,lambda);
[conf yi] = max(gamma');
distses = [distses sum(testlabels(yi)==Xlabels)/(ntest*10)*100];

[C gamma] = OTsolve(cost1,x,y,tol,lambda);
[conf yi] = max(gamma');
concavses = [concavses sum(testlabels(yi)==Xlabels)/(ntest*10)*100];

[C gamma] = OTsolve(cost2,x,y,tol,lambda);
[conf yi] = max(gamma');
convexses = [convexses sum(testlabels(yi)==Xlabels)/(ntest*10)*100];
end
lambdastore = [lambdastore lambda];
convexmean = [convexmean mean(convexses)]; concavmean = [concavmean mean(concavses)];
distmean = [distmean mean(distses)];
convexerror = [convexerror std(convexses)/sqrt(N)];
concaverror = [concaverror std(concavses)/sqrt(N)];
disterror = [disterror std(distses)/sqrt(N)];
convexses = []; concavses = []; distses = [];
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
y = distmean; err = disterror;
patch([x fliplr(x)],[y+err fliplr(y-err)],[0.7 0.9 0.7],'EdgeColor','none','FaceAlpha',.5);
dist = plot(x,y,'g.-');
legend([convex dist concave],'Convex Cost','Distance Cost','Concave Cost')