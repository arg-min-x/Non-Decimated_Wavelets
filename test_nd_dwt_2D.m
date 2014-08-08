clear all;
close all;
clc;

level = 1;
x = double(imread('cameraman.tif')) + 1j*ones(256,256);
% x = randn(256,256) + 1j*randn(256,256);
wnames = {'db1','db1'};
nddwt = nd_dwt_2D(wnames,size(x));

x_trans = nddwt.dec(x,level);
x_recon = nddwt.rec(x_trans);

fprintf('Energy in signal domain = %s \t Energy in Wavelet Domain = %s\n',norm(x(:)), norm((x_trans(:))))
fprintf('Absolute Max Reconstruction Error = %s\n',max(abs((x_recon(:))-x(:))))

figure(1)
clf
coeffs = [x_trans(:,:,1),x_trans(:,:,2);x_trans(:,:,3),x_trans(:,:,4)];
imagesc(abs(coeffs))

figure(2)
clf
subplot(211)
imagesc(abs(x))
title('Original Signal')
colormap(gray)

subplot(212)
imagesc(abs(x_recon))
title('Reconstructed Signal')
colormap(gray)
