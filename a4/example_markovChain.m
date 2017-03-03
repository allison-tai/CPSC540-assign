clear all

load viterbiData.mat

decode_short1 = exactDecode(p0,pT_short1)

decode_short2 = exactDecode(p0,pT_short2)

decode_long = exactDecode(p0,pT_long)
