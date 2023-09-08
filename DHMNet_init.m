function net = DHMNet_init(varargin)
% DMHNet_init initializes a DMHNet

rng('default');
rng(0) ;

opts.layernum = 9;

Winit1 = cell(opts.layernum,1);
opts.datadim = [63, 53, 43, 33];
for iw = 1 : length(opts.datadim)-1
    A = rand(opts.datadim(iw));
    [U1, ~, ~] = svd(A * A');
    Winit1{iw} = U1(:,1:opts.datadim(iw+1));
end
f=1/100 ;
classNum = 45;
fdim1 = size(Winit1{iw},2)*size(Winit1{iw},2);
Winit1{4} = f*randn(fdim1, classNum, 'single');


opts.datadim = [63, 53, 43, 33];
opts.skedim = [8, 8, 8, 8];
opts.pool = [2, 2, 2, 2, 2];
opts.layernum2 = length(opts.datadim)-1;
Winit2 = cell(opts.layernum2+1,1);
for iw = 1 :  opts.layernum2
    for i_s = 1 : opts.skedim(iw)
        A = rand(opts.datadim(iw));
        [U1, ~, ~] = svd(A * A');
        Winit2{iw}(:,:,i_s) = U1(:,1:opts.datadim(iw+1))';
    end
end
fdim2 = opts.datadim(end)*opts.datadim(end)*opts.skedim(end);
Winit2{4}  = f*randn(fdim2, classNum, 'single');


opts.datadim = [33, 43, 33];
for iw = 1 : length(opts.datadim)-1
    if iw == 1
        A = rand(opts.datadim(iw+1));
        [U1, ~, ~] = svd(A * A');
        Winit1{5} = (U1(:,1:opts.datadim(iw)))';
    else
        A = rand(opts.datadim(iw));
        [U1, ~, ~] = svd(A * A');
        Winit1{6} = U1(:,1:opts.datadim(iw+1));
    end
end
Winit1{7} = f*randn(fdim1, classNum, 'single');

opts.datadim = [33, 43, 33];
for iw = 1 : length(opts.datadim)-1
    for i_s = 1 : opts.skedim(iw)
        if iw == 1
            A = rand(opts.datadim(iw+1));
            [U1, ~, ~] = svd(A * A');
            Winit2{5}(:,:,i_s) = (U1(:,1:opts.datadim(iw)));
        else
            A = rand(opts.datadim(iw));
            [U1, ~, ~] = svd(A * A');
            Winit2{6}(:,:,i_s) = U1(:,1:opts.datadim(iw+1))';
        end
    end
end
Winit2{7}  = f*randn(fdim2, classNum, 'single');
Winit3{1}  = f*randn(fdim1+fdim2, classNum, 'single');
Winit3{2}  = f*randn(fdim1+fdim2, classNum, 'single');


A = rand(43);
[U1, ~, ~] = svd(A * A');
Winit1{8} =(U1(:,1:33))';

for i_s = 1 : 8
    A = rand(43);
    [U1, ~, ~] = svd(A * A');
    Winit2{8}(:,:,i_s) = (U1(:,1:33));
end




net.layers = {} ;
net.layers{1} = struct('type', 'bfc', 'weight', Winit1{1}) ; 
net.layers{2} = struct('type', 'rec') ;
net.layers{3} = struct('type', 'bfc', 'weight', Winit1{2}) ; 
net.layers{4} = struct('type', 'rec') ;
net.layers{5} = struct('type', 'bfc', 'weight', Winit1{3}) ;
net.layers{6} = struct('type', 'log') ;
net.layers{7} = struct('type', 'fc_spd', 'weight', Winit1{4}) ;
net.layers{8} = struct('type', 'softmaxloss') ;


net.layers{9} = struct('type', 'orthmap') ;
net.layers{10} = struct('type', 'frmap','weight', Winit2{1}) ;
net.layers{11} = struct('type', 'reorth') ;
net.layers{12} = struct('type', 'frmap','weight', Winit2{2}) ;
net.layers{13} = struct('type', 'reorth') ;
net.layers{14} = struct('type', 'frmap','weight', Winit2{3}) ;
net.layers{15} = struct('type', 'reorth') ;
net.layers{16} = struct('type', 'projmap') ;
net.layers{17} = struct('type', 'fc_gr', 'weight', Winit2{4}) ;
net.layers{18} = struct('type', 'softmaxloss') ;

net.layers{19} = struct('type', 'addFeature') ;
net.layers{20} = struct('type', 'fc', 'weight', Winit3{1}) ;
net.layers{21} = struct('type', 'softmaxloss') ;





net.layers{22} = struct('type', 'rec') ;
net.layers{23} = struct('type', 'bfc', 'weight', Winit1{5}) ; 
net.layers{24} = struct('type', 'rec') ;
net.layers{25} = struct('type', 'bfc', 'weight', Winit1{6}) ; 
net.layers{26} = struct('type', 'log') ;
net.layers{27} = struct('type', 'fc_spd', 'weight', Winit1{7}) ;
net.layers{28} = struct('type', 'softmaxloss') ;


net.layers{29} = struct('type', 'frmap','weight', Winit2{5}) ;
net.layers{30} = struct('type', 'reorth') ;
net.layers{31} = struct('type', 'frmap','weight', Winit2{6}) ;
net.layers{32} = struct('type', 'reorth') ;
net.layers{33} = struct('type', 'projmap') ;
net.layers{34} = struct('type', 'fc_gr', 'weight', Winit2{7}) ;
net.layers{35} = struct('type', 'softmaxloss') ;

net.layers{36} = struct('type', 'addFeature') ;
net.layers{37} = struct('type', 'fc', 'weight', Winit3{2}) ;
net.layers{38} = struct('type', 'softmaxloss') ;


net.layers{39} = struct('type', 'rec') ;
net.layers{40} = struct('type', 'bfc', 'weight', Winit1{8}) ; 

net.layers{41} = struct('type', 'frmap','weight', Winit2{8}) ;
net.layers{42} = struct('type', 'reorth') ;

net.layers{43} = struct('type', 'reconstructionloss') ;
net.layers{44} = struct('type', 'reconstructionloss') ;
net.layers{45} = struct('type', 'reconstructionloss1') ;
net.layers{46} = struct('type', 'reconstructionloss1') ;

net.layers{47} = struct('type', 'MFM') ;
net.layers{48} = struct('type', 'softmaxloss') ;
