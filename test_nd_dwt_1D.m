clear all;
close all;
clc;

load noisbloc;
level = 3;
x = noisbloc.*(ones(size(noisbloc)) + 1j*ones(size(noisbloc)));
% x = randn(1024,1) + 1j*randn(1024,1);

nddwt = nd_dwt_1D('db8',length(x));

x_trans = nddwt.dec(x,level);
x_recon = nddwt.rec(x_trans);

fprintf('Energy in signal domain = %s \t Energy in Wavelet Domain = %s\n',norm(x), norm((x_trans(:))))
fprintf('Absolute Max Reconstruction Error = %s\n',max(abs((x_recon(:))-x(:))))

figure(1)
clf
subplot(211)
plot(abs(x))
title('Original Signal')

subplot(212)
plot(abs(x_recon))
title('Reconstructed Signal')


figure(2)
subplot(311)
plot(abs(x_trans(:,1)))
title('Approximate Coefficients')

subplot(312)
plot(abs(x_trans(:,2)))
title('Detail Coefficients')

subplot(313)
plot(abs(x_trans(:,3)))
title('Detail Coefficients')
