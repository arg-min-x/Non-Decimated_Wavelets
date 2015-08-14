clear all

x = randn(127,128,32) + 1j*randn(127,128,32);
k = randn(127,128,32) + 1j*randn(127,128,32);


nd_dwt_mex(x,k);