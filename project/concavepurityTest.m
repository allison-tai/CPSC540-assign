clear all
load MNIST.mat

concavses = []; convexses = []; distses = [];
convexmean = []; concavmean = []; distmean = [];
convexerror = []; concaverror = []; disterror = [];
var = [];
N = 150;
for index = 0:4
for i = 1:N
X = images;
n = index; % number of random samples/10
ntest = 10; % number of total samples

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
    I = [I length(find(Xlabels==i))];
    Itest = [Itest length(find(testlabels==i))];
end
var = [var norm(I-Itest,1)/2];

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
end
x = [];
for i = 0:max(var)
    I = find(var==i);
    N = sqrt(length(I));
    if length(I)>1
        x = [x 100-i/(ntest)*10];
        convexmean = [convexmean mean(convexses(I))];
        convexerror = [convexerror std(convexses(I))/N];
        distmean = [distmean mean(distses(I))];
        disterror = [disterror std(distses(I))/N];
        concavmean = [concavmean mean(concavses(I))];
        concaverror = [concaverror std(concavses(I))/N];
    end
end
figure; hold on; y = convexmean; err = convexerror;
set(gca,'FontSize', 20);
title('Concave vs. Convex cost against Similarity');
xlabel('% Similarity'); ylabel('% Accuracy');
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