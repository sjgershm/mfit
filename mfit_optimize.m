function results = mfit_optimize(likfun,param,data,nstarts)
    
    % Find maximum a posteriori parameter estimates.
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
    results.K = K;
    
    % save info to results structure
    results.param = param;
    results.likfun = likfun;
    
    % extract lower and upper bounds
    lb = [param.lb];
    ub = [param.ub];
    
    options = optimset('Display','off');
    warning off all
    
    for s = 1:length(data)
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
            if i == 1 || results.logpost(s) < logp
                results.logpost(s) = logp;
                results.loglik(s) = likfun(x,data(s));
                results.x(s,:) = x;
                results.H{s} = NumHessian(f,x);
            end
        end
        
        results.bic(s,1) = K*log(data(s).N) - 2*results.loglik(s);
        results.aic(s,1) = K*2 - 2*results.loglik(s);
        try
            [~,results.latents(s)] = likfun(results.x(s,:),data(s));
        catch
            results.latents = [];
        end
    end