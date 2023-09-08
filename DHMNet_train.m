function [net, info] = DMHNet_train(net, spd_train, opts)
opts.errorLabels = {'top1e'};
opts.train = find(spd_train.spd.set==1) ;
opts.val = find(spd_train.spd.set==2) ;
for epoch=1745:1745
     % fast-forward to last checkpoint
     modelPath = @(ep) fullfile(opts.dataDir, sprintf('net-epoch-%d.mat', ep));
     modelFigPath = fullfile(opts.dataDir, 'net-train.pdf') ;
             fprintf('resuming by loading epoch %d\n', epoch-1) ;
             load(modelPath(epoch-1), 'net', 'info') ;
             mins = info.('val').max_acc;
             epocheasss = info.('val').max_acc_epoch;
    val = opts.val;
    [net,stats.val, mins,epocheasss] = process_epoch(opts, epoch, spd_train, val, 0, net, mins,epocheasss) ;
    fprintf('The accuracy of the classification results is %.4f %%', mins*100);
end

function [net,stats, mins,epocheasss] = process_epoch(opts, epoch, spd_train, trainInd, learningRate, net, mins,epocheasss)

training = learningRate > 0 ;
if training, mode = 'training' ; else mode = 'validation' ; end

stats = [0 ; 0 ; 0];
numGpus = numel(opts.gpus) ;
if numGpus >= 1
    one = gpuArray(single(1)) ;
else
    one = single(1) ;
end

batchSize = opts.batchSize;
errors = 0;
numDone = 0 ;

for ib = 1 : batchSize : length(trainInd)
    fprintf('%s: epoch %02d: batch %3d/%3d:', mode, epoch, ib,length(trainInd)) ;
    batchTime = tic ;
    res = [];
    if (ib+batchSize> length(trainInd))
        batchSize_r = length(trainInd)-ib+1;
    else
        batchSize_r = batchSize;
    end
    spd_data = cell(batchSize_r,1);    
    spd_label = zeros(batchSize_r,1);
    for ib_r = 1 : batchSize_r
        spdPath = [spd_train.SpdDir '\' spd_train.spd.name{trainInd(ib+ib_r-1)}];
        load(spdPath); spd_data{ib_r} = temp_2;
        spd_label(ib_r) = spd_train.spd.label(trainInd(ib+ib_r-1));

    end
    net.layers{48}.class = spd_label ;
    net.layers{38}.class = spd_label ;
    net.layers{35}.class = spd_label ;
    net.layers{28}.class = spd_label ;
    net.layers{21}.class = spd_label ;
    net.layers{18}.class = spd_label ;
    net.layers{8}.class = spd_label ;
    
    %forward
    if training, dzdy = one; else dzdy = [] ; end
    res = vl_myforbackward(net, spd_data, dzdy, res) ;
  
    % accumulate training errors
    predictions = gather(res(end-1).x) ;
    [~,pre_label] = sort(predictions, 'descend') ;
    error = sum(~bsxfun(@eq, pre_label(1,:)', spd_label)) ;
    numDone = numDone + batchSize_r ;
    errors = errors+error;
    batchTime = toc(batchTime) ;
    speed = batchSize/batchTime ;
    stats = stats+[batchTime ; res(end).x ; error;]; % works even when stats=[] 
    fprintf(' %.2f s (%.1f data/s)', batchTime, speed) ;
    fprintf(' acc: %.5f%%', 1-stats(3)/numDone) ;
    fprintf(' [%d/%d]', numDone, batchSize_r);
    fprintf('\n') ;
end