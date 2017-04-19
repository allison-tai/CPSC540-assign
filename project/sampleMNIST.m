function [image label] = sampleMNIST(n,images,labels)
%SAMPLEMNIST gets n random representations of each number 0-10 in the MNIST dataset 
    Isamples = []; %indices
    for i = 0:9
        I = find(labels==i);
        Isamples = [Isamples I(randsample(length(I),n))];
    end
    image = images(:,:,Isamples); 
    label = labels(Isamples); 
    label = reshape(label,[10*n 1]);
end

