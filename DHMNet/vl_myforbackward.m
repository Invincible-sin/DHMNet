function res = vl_myforbackward(net, x, dzdy, res, varargin)
% vl_myforbackward  evaluates a simple SPDNet
opts.res = [] ;
opts.conserveMemory = false ;
opts.sync = false ;
opts.disableDropout = false ;
opts.freezeDropout = false ;
opts.accumulate = false ;
opts.cudnn = true ;
opts.skipForward = false;
opts.backPropDepth = +inf ;
opts.epsilon = 1e-5;
opts.p = 10;
% opts = vl_argparse(opts, varargin);
gamma = 0.1;
wei = 0.001;
n = numel(net.layers) ;

if (nargin <= 2) || isempty(dzdy)
  doder = false ;
else
  doder = true ;
end

if opts.cudnn
  cudnn = {'CuDNN'} ;
else
  cudnn = {'NoCuDNN'} ;
end

gpuMode = isa(x, 'gpuArray') ;

if nargin <= 3 || isempty(res)
  res = struct(...
    'x', cell(1,n+1), ...
    'dzdx', cell(1,n+1), ...
    'dzdw', cell(1,n+1), ...
    'aux', cell(1,n+1), ...
    'time', num2cell(zeros(1,n+1)), ...
    'backwardTime', num2cell(zeros(1,n+1))) ;
end
if ~opts.skipForward
  res(1).x = x ;
end


% -------------------------------------------------------------------------
%                                                              Forward pass
% -------------------------------------------------------------------------

for i=1:n
  if opts.skipForward, break; end;
  l = net.layers{i} ;
  res(i).time = tic ;
  switch l.type
    case 'bfc'
      res(i+1).x = vl_mybfc(res(i).x, l.weight) ; 
    case 'fc'
      res(i+1).x = vl_myfc(res(i).x, l.weight) ; 
    case 'fc_spd'
      res(i+1).x = vl_myfc_spd(res(i).x, l.weight) ; 
    case 'fc_gr'
      res(i+1).x = vl_myfc_gr(res(i).x, l.weight) ;    
    case 'rec'
        if doder
            alt = doder;
        else
            alt = doder + 1;
        end
        if i==22
           [res(i+1).x, res(i+1).SS] = vl_myrec(res(6).x, opts.epsilon, alt, []) ;
        elseif i==39
           [res(i+1).x, res(i+1).SS] = vl_myrec(res(26).x, opts.epsilon, alt, []) ;
        else
          [res(i+1).x, res(i+1).SS] = vl_myrec(res(i).x, opts.epsilon, alt, []) ;
        end
    case 'log'
        if doder
            alt = doder;
        else
            alt = doder + 1;
        end
        [res(i+1).x, res(i+1).SS] = vl_mylog(res(i).x, alt, []) ;
    case 'orthmap'
       if(i==9)
        [res(i+1).x, ~] = vl_myorthmap(res(1).x, opts.p) ; 
       else
        [res(i+1).x, res(i)] = vl_myorthmap(res(i), opts.p) ; 
       end
    case 'frmap'
        if i==29
          res(i+1).x = vl_myfrmap(res(16).x, l.weight) ;    
        elseif i==41
          res(i+1).x = vl_myfrmap(res(33).x, l.weight) ;    
        else
          res(i+1).x = vl_myfrmap(res(i).x, l.weight) ;    
        end
    case 'reorth'
      [res(i+1).x, res(i)] = vl_myreorth(res(i)) ;      
    case 'projmap'
      res(i+1).x = vl_myprojmap(res(i).x) ;   
    case 'addFeature'
     if i==19
        res(i+1).x = vl_myaddFeature(res(7).x, res(17).x) ;  
     else
        res(i+1).x = vl_myaddFeature(res(i-9).x, res(i-2).x) ;  
     end
  case 'MFM'
      res(i+1).x = vl_myaddMFM(res(21).x, res(38).x) ;  
   case 'reconstructionloss'
      if i == 43
          data_hou = res(24).x; 
          data_chu = res(4).x;
      else
          data_hou = res(41).x; 
          data_chu = res(24).x;
      end
      res(i+1).obj = vl_myreconstructionloss(data_hou, data_chu, gamma); 
      res(i+1).x = res(i).x;
   case 'reconstructionloss1'
      if i == 45
          data_hou = res(31).x; 
          data_chu = res(14).x;
      else 
          data_hou = res(43).x; 
          data_chu = res(31).x;
      end
      res(i+1).obj = vl_myreconstructionloss1(data_hou, data_chu, gamma); 
      res(i+1).x = res(i).x;
  case 'softmaxloss'
      res(i+1).x = vl_mysoftmaxloss(res(i).x, l.class) ; 
    case 'custom'
      res(i+1) = l.forward(l, res(i), res(i+1)) ;
    otherwise
      error('Unknown layer type %s', l.type) ;
  end
  % optionally forget intermediate results
  forget = opts.conserveMemory ;
  forget = forget & (~doder || strcmp(l.type, 'relu')) ;
  forget = forget & ~(strcmp(l.type, 'loss') || strcmp(l.type, 'softmaxloss')) ;
  forget = forget & (~isfield(l, 'rememberOutput') || ~l.rememberOutput) ;
  if forget
    res(i).x = [] ;
  end
  if gpuMode & opts.sync
    % This should make things slower, but on MATLAB 2014a it is necessary
    % for any decent performance.
    wait(gpuDevice) ;
  end
  res(i).time = toc(res(i).time) ;
end