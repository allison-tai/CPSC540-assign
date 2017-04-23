function [xnew] = pixelTrans(gamma, x, n)
% translates pixels of x by average move

[g1,g2,g3] = size(gamma);

distX = zeros(g1, g3);
distY = zeros(g1, g3);
    
for i = 1:g1
    % look through each pixel
    currX = mod(i,n);
    currY = ceil(i/n);
        
    for j = 1:g3
        % look through each test set
        % get the 28 x 28 thing
        distribution = reshape(gamma(i,:,j), [n n]);
        % get average point (expectation)
        bestX = round(sum((1:n).*(mean(distribution, 1)./sum(mean(distribution, 1)))));
        bestY = round(sum((1:n).*(mean(distribution, 2)./sum(mean(distribution, 2)))'));
        % now find how many pixels vertically/horizontally we need to move
        distX(i,j) = bestX - currX;
        distY(i,j) = bestY - currY;
    end
end

distX(isnan(distX)) = 0;
distY(isnan(distY)) = 0;
% mean move of all pixels (we'll shift every pixel in the image by this
% much
moveX = round(mean((mean(distX))~=0));
moveY = round(mean((mean(distY))~=0));
% now shift
xsmall = reshape(x, [n n]);
xnew = zeros(n, n);
xnew(moveX+1:end,moveY+1:end) = xsmall(1:end-moveX,1:end-moveY);
xnew = reshape(xnew, [g1 1]);

end