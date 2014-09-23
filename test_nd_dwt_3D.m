clear all;
close all;
clc;

sizes = [64,64,20];
x = randn(sizes) + 1j*randn(sizes);
wnames = {'db8','db3','db9'};
% wnames = 'db10';
dwt = nd_dwt_3D(wnames,size(x),1);

x_trans = dwt.dec(x,3);
x_recon = dwt.rec(x_trans);

fprintf('Energy in signal domain = %s \t Energy in Wavelet Domain = %s\n',norm(x(:)), norm((x_trans(:))))
fprintf('Absolute Max Reconstruction Error = %s\n',max(abs((x_recon(:))-x(:))))
