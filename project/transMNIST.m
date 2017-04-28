clear all
load MNIST.mat

% how many images will we translate? All of them, I guess
nSamples = 60000;
% size of each image
n = size(images,1);

%v = randi(60000, [1 nSamples]);
%images = images(:,:,v);
transImages = zeros(n, n, nSamples);

for i = 1:nSamples
    currOld = images(:,:,i);
    currNew = zeros(n,n);
    moveX = randi([-7, 7], [1 1]);
    moveY = randi([-4, 4], [1 1]);
    if (moveX >= 0)
        currNew(moveX+1:end,:) = currOld(1:end-moveX,:);
    else 
        currNew(1:end+moveX,:) = currOld(1-moveX:end,:);
    end
    if (moveY >= 0)
        currNew(:,moveY+1:end) = currOld(:,1:end-moveY);
    else
        currNew(:,1:end+moveY) = currOld(:,1-moveY:end);
    end
    transImages(:,:,i) = currNew;
end

save('transMNIST.mat', 'transImages', 'labels');