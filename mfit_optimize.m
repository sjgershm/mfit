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
    % Sam Gershman, March 2019
    
    % fill in missing options
    if nargin < 4 || isempty(nstarts); nstarts = 5; end
    K = length(param);
    results.K = K;
    
    % save info to results structure
    results.param = param;
    results.likfun = likfun;
    
    % extract lower and upper bounds
    if ~isfield(param,'lb'); lb = zeros(size(param)) + -inf; else lb = [param.lb]; end
    if ~isfield(param,'ub'); ub = zeros(size(param)) + inf; else ub = [param.ub]; end
    
    options = optimset('Display','off','MaxFunEvals',2000);
    warning off all
    
    if isfield(param,'x0'); nstarts = length(param(1).x0); end
    
    for s = 1:length(data)
        disp(['Subject ',num2str(s)]);
        
        % construct posterior function
        f = @(x) -mfit_post(x,param,data(s),likfun);
        
        for i = 1:nstarts
            if all(isinf(lb)) && all(isinf(ub))
                if isfield(param,'x0')
                    for j = 1:length(param)
                        x0(j) = param(j).x0(i);
                    end
                else
                    x0 = randn(1,K);
                end
                [x,nlogp,~,~,~,H] = fminunc(f,x0,options);
            else
                if isfield(param,'x0')
                    for j = 1:length(param)
                        x0(j) = param(j).x0(i);
                    end
                else
                    x0 = zeros(1,K);
                    for k = 1:K
                        x0(k) = unifrnd(param(k).lb,param(k).ub);
                    end
                end
                [x,nlogp,~,~,~,~,H] = fmincon(f,x0,[],[],[],[],lb,ub,[],options);
            end
            logp = -nlogp;
            if i == 1 || results.logpost(s) < logp
                results.logpost(s) = logp;
                results.loglik(s) = likfun(x,data(s));
                results.x(s,:) = x;
                results.H{s} = H;
            end
        end
        
        results.bic(s,1) = K*log(data(s).N) - 2*results.loglik(s);
        results.aic(s,1) = K*2 - 2*results.loglik(s);
        try
            [~,results.latents(s)] = likfun(results.x(s,:),data(s));
        catch
            disp('no latents')
            results.latents = [];
        end
    end