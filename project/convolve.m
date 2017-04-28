function [ imgfilt ] = convolve(img,filter,stride)
%CONVOLVE Convolves img with filter, with stride
    img = reshape(img,[28 28]);
    [m n] = size(filter);
    [M N] = size(img);
    imgfilt = zeros(M,N);
    newimage = zeros(30,30); % add zeros to edge of image
    newimage(2:29,2:29) = img;
    img = newimage;
    for i = 2:M+1
        for j = 2:M+1
            imgfilt(i,j) = sum(sum(filter.*img(i-1:i+m-2,j-1:j+n-2)));
        end
    end
    % make everything positive
    %imgfilt =abs(imgfilt)/max(max(abs(imgfilt))); 
    imgfilt =(imgfilt+min(min(imgfilt)))/max(max(imgfilt)); 
    imgfilt = reshape(imgfilt(2:29,2:29),[28^2 1]);
end

