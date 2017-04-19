function [ imgfilt ] = convolve(img,filter,stride)
%CONVOLVE Convolves img with filter, with stride
    [m n] = size(filter);
    [M N] = size(img);
    mu = floor(M-m+1/stride);
    nu = floor(N-n+1/stride);
    imgfilt = zeros(mu,nu);
    for i = 1:mu
        for j = 1:nu
            imgfilt(i,j) = sum(sum(filter.*img(i:i+m-1,j:j+n-1)));
        end
    end
    % make everything positive
    %imgfilt =abs(imgfilt)/max(max(abs(imgfilt))); 
    imgfilt =(imgfilt+min(min(imgfilt)))/max(max(imgfilt)); 
end

