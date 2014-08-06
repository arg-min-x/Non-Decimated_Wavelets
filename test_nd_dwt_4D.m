clear all;
close all;
clc;

x = randn(64,64,20,16) + 1j*randn(64,64,20,16);
level = 1;
wnames = {'db3','db3','db1','db3'};
dwt = nd_dwt_4D('db3',size(x));

x_trans = dwt.dec(x,level);
x_recon = dwt.rec(x_trans);

fprintf('Energy in signal domain = %s \t Energy in Wavelet Domain = %s\n',norm(x(:)), norm((x_trans(:))))
fprintf('Absolute Max Reconstruction Error = %s\n',max(abs((x_recon(:))-x(:))))



