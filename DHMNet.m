function [net, info] = DMHNet(varargin)
%set up the path
confPath;
%parameter setting
opts.dataDir = fullfile('./data/afew') ;
opts.imdbPathtrain = fullfile(opts.dataDir, 'SPD_info.mat');
opts.batchSize = 30 ;
opts.test.batchSize = 1;
opts.gpus = [] ;
opts.continue = 1;
%spdnet initialization
net = DHMNet_init() ;
%loading metadata 
load(opts.imdbPathtrain) ;
%spdnet training
[net, info] = DHMNet_train(net, spd_train, opts);
