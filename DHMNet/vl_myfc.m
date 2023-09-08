function [Y, Y_w] = vl_myfc(X, W, dzdy)
%fully-connected convolutional (FC) layer

if nargin < 3
    Y = W' * X;
else
    Y = W * dzdy;
    Y_w = X*dzdy';
end