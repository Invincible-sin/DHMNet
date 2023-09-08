function Y = vl_myreconstructionloss1(X, X_ori, gamma)
  % this function is designed to implement the decode term with the reconstruction function
  % Date:
  % Author: 
  % Coryright@

  dzdy_l3 = single(1);
  %gamma = 0.1; % needs to be adjusted 0.01
  [n1,n2,len,chan] = size(X);
  dist_sum = zeros(len,chan); % save each pair dist
  Y = zeros(n1,n2,len,chan); % save obj or dev
  dev_term = zeros(n1,n2,len,chan); % save each pair' derivation 
  for i = 1 : len
      for channel = 1:chan
          temp = X(:,:,len,channel) - X_ori(:,:,len,channel);%100,10
          dev_term(:,:,i,channel) = 2 * temp;
          dist_sum(i,channel) = norm(temp,'fro') * norm(temp,'fro'); % the dist computed via LEM
      end
  end
  if nargin < 3
      Y = gamma * (sum(dist_sum(:)) / (len*chan)); % the obj og this loss function
  else
      for ix = 1 : len
          for ichannel =1:chan 
              dev_l3 = bsxfun(@times, dev_term(:,:,ix,ichannel), bsxfun(@times, ones(n1,n2), dzdy_l3));
              Y(:,:,ix,ichannel) = gamma * dev_l3; % the sum of rebuild term and softmax term, then push them into margin term
          end
      end
  end
end

