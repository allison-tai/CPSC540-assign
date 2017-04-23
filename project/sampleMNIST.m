function [image label] = sampleMNIST(n,images,labels,Index)
%SAMPLEMNIST gets n random representations of each number in Index in the MNIST dataset 
    Isamples = []; %indices
    for i = Index
        I = find(labels==i);
        Isamples = [Isamples I(randsample(length(I),n))];
    end
    image = images(:,:,Isamples); 
    label = labels(Isamples); 
    label = reshape(label,[length(Index)*n 1]);
end

