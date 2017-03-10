function [yOpt] = exactDecode(p0,pT)
% Exact decoding of (short) Markov binary chains

nNodes = size(pT,3)+1;

% Initial value of states to try
y = ones(nNodes,1);

maxProb = -inf;
while 1
    % Evaluate this sequence of states
    prob = decodeProb(y,p0,pT);
    
    % Check is this sequence is the most likely
    if prob > maxProb
        maxProb = prob;
        yOpt = y;
    end
    
    % Move on to next sequence to try
    y = nextSequence(y);
    if all(y==1)
        % We are back at the first sequence
        break;
    end
end
end

function [y] = nextSequence(y)
nNodes = length(y);
for ind = 1:nNodes
    if y(ind) == 1
        y(ind) = 2;
        break;
    else
        y(ind) = 1;
        
    end    
end
end