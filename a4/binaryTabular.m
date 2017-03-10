function [model] = binaryTabular(X,y,alpha)
% Predicts binary {0,1} y from binary {0,1}^d X by memorizing training
% data frequenceies, and Laplace smooothing by alpha

[n,d] = size(X);

% Find unique rows and their indices
[rows,~,ind] = unique([X y],'rows');

% Store frequency of each unique row
counts = histc(ind,1:max(ind));

model.rows = rows;
model.counts = counts;
model.sample = @sample;
model.alpha = alpha;
end

function [ysample] = sample(model,xhat)
rows = model.rows;
counts = model.counts;
alpha = model.alpha;

nRows = size(rows,1);
ind = find(all(model.rows == repmat([xhat 0],[nRows 1]),2));
if isempty(ind)
    p0 = model.alpha;
else
    p0 = model.alpha + counts(ind);
end
ind = find(all(model.rows == repmat([xhat 1],[nRows 1]),2));
if isempty(ind)
    p1 = alpha;
else
    p1 = alpha + counts(ind);
end
p1/(p0+p1)
ysample = rand < p1/(p0+p1);
end