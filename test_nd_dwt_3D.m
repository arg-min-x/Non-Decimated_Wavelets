clear all;
close all;
clc;

x = randn(123,139,27) + 1j*randn(123,139,27);

dwt = nd_dwt_3D({'db8','db3'},size(x));

x_trans = dwt.dec(x,2);
x_recon = dwt.rec(x_trans);

norm(x_trans(:))
norm(x(:))
% 
max(abs(x_recon(:)-x(:)))
% 
% 
% tic;
% for ind = 1:1000
%     tmp = x.*conj(dwt.f_dec.LLL);
% end
% toc
% 
% tic;
% for ind = 1:1000
%     tmp = x.*dwt.f_dec.LLL;
% end
% toc

