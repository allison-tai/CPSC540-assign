%%
clear all
close all
load PINS.mat

[n,d] = size(X);
k = 10; % We have 10 digits

% Example of computing marginal of first PIN number being 1
p_11 = sum(X(:,1)==1)/n

%% Make Adjacency Matrix and EdgeStruct
adj = zeros(d); % First lets try an empty adjacency matrix
edgeStruct = UGM_makeEdgeStruct(adj,k);

%% Choose parameter tieing scheme
ising = 0; % Don't use Ising potentials
tied = 1; % Use tied node/edge parameters
[nodeMap,edgeMap,w] = UGM_makeMRFmaps(edgeStruct,ising,tied);

%% Compute sufficient statistics
suffStat = UGM_MRF_computeSuffStat(X,nodeMap,edgeMap,edgeStruct);

%% Train a UGM that matches the sufficient statistics
options = []; % Optimization options
inferFunc = @UGM_Infer_Exact; % Function used to perform inference in the model
w = minFunc(@UGM_MRF_NLL,w,options,n,suffStat,nodeMap,edgeMap,edgeStruct,inferFunc);

%% Form the estimated potentials and do decoding/inference/sampling with them

% Compute node and edge potentials
[nodePot,edgePot] = UGM_MRF_makePotentials(w,nodeMap,edgeMap,edgeStruct);

% Do decoding in the model
xDecode = UGM_Decode_Exact(nodePot,edgePot,edgeStruct);

% Compute node and edge marginals, as well as normalizing constant
[nodeMarg,edgeMarg,logZ] = UGM_Infer_Exact(nodePot,edgePot,edgeStruct);
Z = exp(logZ);

% Generate samples
samples = UGM_Sample_Exact(nodePot,edgePot,edgeStruct)';

% Do conditional inference assuming first three numbers are {1,2,3}
clamped = [1 2 3 0];
[condNP,condEP,condES,condEM] = UGM_makeClampedPotentials(nodePot,edgePot,edgeStruct,clamped);
[condNM,condEM,conLogZ] = UGM_Infer_Exact(condNP,condEP,condES);

