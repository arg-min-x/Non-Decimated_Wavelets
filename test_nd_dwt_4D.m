clear all;
close all;
clc;

sizes = [64,64,20,16];
x = randn(sizes) + 1j*randn(sizes);
level = 2;
% wnames = {'db3','db3','db1','db3'};
wnames = {'db4','db3','db1','db5'};
dwt = nd_dwt_4D(wnames,size(x));

x_trans = dwt.dec(x,level);
x_recon = dwt.rec(x_trans);

fprintf('Energy in signal domain = %s \t Energy in Wavelet Domain = %s\n',norm(x(:)), norm((x_trans(:))))
fprintf('Absolute Max Reconstruction Error = %s\n',max(abs((x_recon(:))-x(:))))

