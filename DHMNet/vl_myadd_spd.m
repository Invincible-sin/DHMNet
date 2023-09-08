function [X_sc1,X_sc2,X_sc3] = vl_myadd_spd(X_sc1, X_sc2, X_sc3, dzdy)

x_size =  size(X_sc1{1},1);
len = length(X_sc1);
X_t = zeros(3*x_size^2,len);
for ix = 1 : len
 X_t(1+0*x_size^2:1*x_size^2,ix) = X_sc1{ix}(:);
 X_t(1+1*x_size^2:2*x_size^2,ix) = X_sc2{ix}(:);
 X_t(1+2*x_size^2:3*x_size^2,ix) = X_sc3{ix}(:);   
end
if nargin < 4
   X_sc1 = X_t;
else
   X_sc1 = dzdy(1+0*x_size^2:1*x_size^2,:);
   X_sc2 = dzdy(1+1*x_size^2:2*x_size^2,:);
   X_sc3 = dzdy(1+2*x_size^2:3*x_size^2,:);
end
