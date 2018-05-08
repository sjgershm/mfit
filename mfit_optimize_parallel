function results = mfit_optimize_parallel(likfun,param,data,nstarts)

% Find maximum a posteriori parameter estimates. Parallelized for effeciency.
%
% USAGE: results = mfit_optimize(likfun,param,data,[nstarts])
%
% INPUTS:
%   likfun - likelihood function handle
%   param - [K x 1] parameter structure
%   data - [S x 1] data structure
%   nstarts (optional) - number of random starts (default: 5)
%
% OUTPUTS:
%   results - structure with the following fields:
%               .x - [S x K] parameter estimates
%               .logpost - [S x 1] log posterior
%               .loglik - [S x 1] log likelihood
%               .bic - [S x 1] Bayesian information criterion
%               .aic - [S x 1] Akaike information criterion
%               .H - [S x 1] cell array of Hessian matrices
%               .latents - latent variables (only if likfun returns a second argument)
%
% Sam Gershman, July 2017

% fill in missing options
if nargin < 4 || isempty(nstarts); nstarts = 5; end
K = length(param);
S = length(data);
results.K = K;
results.S = S;

% save info to results structure
results.param = param;
results.likfun = likfun;

% extract lower and upper bounds
lb = [param.lb];
ub = [param.ub];

% preallocate
[logpost, loglik, bic, aic] = deal(nan(S, 1));
[x_out] = nan(S, K);
[H] = cell(S, 1);

options = optimset('Display','off');
warning off all

parfor s = 1:S
    
    disp(['Subject ',num2str(s)]);
    
    % construct posterior function
    f = @(x) -mfit_post(x,param,data(s),likfun);
    
    for i = 1:nstarts
        x0 = zeros(1,K);
        for k = 1:K
            x0(k) = unifrnd(param(k).lb,param(k).ub);
        end
        [x,nlogp] = fmincon(f,x0,[],[],[],[],lb,ub,[],options);
        logp = -nlogp;
        if i == 1 || logpost(s) < logp
            logpost(s) = logp;
            loglik(s) = likfun(x,data(s));
            x_out(s,:) = x;
            H{s} = NumHessian(f,x);
        end
    end
    
    bic(s) = K*log(data(s).N) - 2*loglik(s);
    aic(s) = K*2 - 2*loglik(s);
  
end

% save
results.logpost = logpost';
results.loglik = loglik;
results.bic = bic;
results.aic = aic;
results.x = x_out;
results.H = H;

% if likfun returns a 2nd argument, save latent variables
try
    for s = 1:S
        [~, results.latents(s)] = likfun(x_out(s,:),data(s));
    end
catch
    results.latents = [];
end
