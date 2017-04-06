clear all

load viterbiData.mat

decode_short1 = exactDecode(p0,pT_short1)
decode_short1 = sampleBackwards_allison(p0,pT_long)

decode_short2 = sampleBackwardsMC([1 0],pT_long)

%decode_long = exactDecode(p0,pT_long)
