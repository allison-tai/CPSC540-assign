function filterold = cnn(cost,X,Y,Xlabels,Ylabels)
%CNN train cnn
    % convolve x,y
    % pool?
    % transport
    % backprop
filterold = zeros(3,3); filterold(2,2)=1; % initialize identity convolution

% parameters
epsilon = 0.001*ones(3); % numerical approximation step size
alpha = 0.2; % gradient descent step size
tol = 0.001; % update precision

i=1; I = 1:9; % initialize counter and possible steps

M = size(X,3); N = size(Y,3); % number of samples

% calculate initial loss
c0 = zeros(M,N);
tic
for j = 1:M
    yc = squeeze(reshape(Y,[28^2 1 N]));
    [c0(j,:) gamma] = OTsolve(cost,reshape(X(:,:,j),[28^2 1]),yc,0.001,1);
end
old_loss = loss(c0,Xlabels,Ylabels);

% perform gradient descent
while 1
    if i==1 
        % start a new gradient descent
        ei = I(randsample(length(I),1));
        fprintf('index done:\t\t\t %d\nnumber of elements left:\t %d\n',ei,length(I))
    else
        % try the other direction
        epsilon(ei) = -epsilon(ei);
    end
    % create new filter by perturbing old one
    filter = filterold; 
    filter(ei) = filter(ei)+epsilon(ei);
    
    %convolution step
    for j = 1:N
        yc(:,j) = convolve(Y(:,:,j),filter,1);
    end
    for j = 1:M
        xc(:,j) = convolve(X(:,:,j),filter,1);
        [c0(j,:) gamma] = OTsolve(cost,xc(:,j),yc,0.001,1);
    end
    % calculate loss
    new_loss = loss(c0,Xlabels,Ylabels);
    
    
    if new_loss<old_loss-tol
        -new_loss+old_loss
        % gradient descent
        filterold(ei) = filterold(ei)-alpha*max(new_loss-old_loss,-abs(epsilon(ei)))/epsilon(ei);
        fprintf('Changing index %d %d',ceil(ei/3),mod(ei,3))
        i = 1; I = 1:9; % reset counter, array
        % update variables
        old_loss = new_loss;
        % add linesearch step
    elseif i==2
        I = setdiff(I,ei); % if neither direction improves cost remove this index
        i = 1;
        if length(I) == 0
            break % no more indices to improve
        end
    else
        i = i+1; % try the other direction
    end
    %toc
    
 end
toc
end

function loss = loss(c0,Xlabels,Ylabels)
    loss = 0;
    N = sum(sum(c0));
    for i=0:9
        I = find(Xlabels==i);
        J = find(Ylabels==i);
        loss = loss + sum(sum(c0(I,J)))/N;
    end
end
