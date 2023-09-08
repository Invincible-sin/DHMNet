function [SPD, Gr] = vl_myaddFeature(X_spd, X_gr,  dzdy)
%[DZDX, DZDF, DZDB] = vl_myconv(X, F, B, DZDY)
%regular fully connected layer

[n1,n2,~,n4] = size(X_gr);
n_spd = size(X_spd{1},1)^2;
n_all = size(X_spd{1},1)^2+n1*n2*n4;
if nargin < 3
    SPD = zeros(n_all,length(X_spd));
    for ix = 1 : length(X_spd)
        x_t_spd = X_spd{ix};
        x_t_gr = X_gr(:,:,ix,:);
        SPD(1:n_spd,ix) = x_t_spd(:);
        SPD(n_spd+1:n_all,ix) = x_t_gr(:);
    end  
else
    SPD = zeros(n_spd,length(X_spd));
    Gr = zeros(n1*n2*n4,length(X_spd));
    for ix = 1 : length(X_spd)
        SPD(:,ix) = dzdy(1:n_spd,ix);
        Gr(:,ix) = dzdy(n_spd+1:n_all,ix);
    end
end



