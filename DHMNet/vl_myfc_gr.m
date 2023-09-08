function [Y, Y_w] = vl_myfc_gr(X, W, dzdy)
%fully-connected convolutional (FC) layer

[n1,n2,n3,n4] = size(X);

X_t = zeros(n1*n2*n4,n3);

for ix = 1 : n3
    x_t = X(:,:,ix,:);
    X_t(:,ix) = x_t(:);
end
if nargin < 3
    Y = W' * X_t;
else
    Y = W * dzdy;
    Y_w = X_t*dzdy';
end