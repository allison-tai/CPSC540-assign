function [Loss errors] = lossEMD(xi,samplabel,testlabel)
    % calculate loss function from sample values and test values
    error = samplabel(xi) ~= testlabel;
    errors = find(error);
    Loss = sum(error);
end