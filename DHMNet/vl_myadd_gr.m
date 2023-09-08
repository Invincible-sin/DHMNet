function [X_sc1,X_sc2,X_sc3] = vl_myadd_gr(X_sc1, X_sc2, X_sc3, dzdy)
[n1,n2,len,n4] = size(X_sc1);
X_t = zeros(n1*n2*n4*3,len);
for ix = 1 : len
    te = X_sc1(:,:,ix,:);
    X_t(1+0*n1*n2*n4:1*n1*n2*n4,ix) = te(:);
    te = X_sc2(:,:,ix,:);
    X_t(1+1*n1*n2*n4:2*n1*n2*n4,ix) = te(:);
    te = X_sc3(:,:,ix,:);
    X_t(1+2*n1*n2*n4:3*n1*n2*n4,ix) = te(:);   
end
if nargin < 4
   X_sc1 = X_t;
else
    X_sc1 = dzdy(1+0*n1*n2*n4:1*n1*n2*n4,:);
    X_sc2 = dzdy(1+1*n1*n2*n4:2*n1*n2*n4,:);
    X_sc3 = dzdy(1+2*n1*n2*n4:3*n1*n2*n4,:);
end

