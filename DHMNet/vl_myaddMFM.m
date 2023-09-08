function [X_out1, X_out2] = vl_myaddMFM(X1, X2, dzdy)
%[DZDX, DZDF, DZDB] = vl_myconv(X, F, B, DZDY)
%regular fully connected layer
X_out=max(X1, X2);
if nargin < 3
    X_out1 = X_out;
else 
    d1 = X_out-X1;
    d1(d1==0)=1;
    d1(d1~=1)=0;
    maxX = max(X1);
    X1(X1~=maxX)=0;
    X1(X1==maxX)=1;
    d1 = bitand(uint8(X1),uint8(d1));    
    
    d2 = X_out-X2;
    d2(d2==0)=1;
    d2(d2~=1)=0;
    maxX = max(X2);
    X2(X2~=maxX)=0;
    X2(X2==maxX)=1;
    d2 = bitand(uint8(X2),uint8(d2));
     
    X_out1 = double(d1).*dzdy;
    X_out2 = double(d2).*dzdy;
end
