function results = mfit_optimize_hierarchical(likfun,param,data,nstarts,parallel)
    
    % Hierarchical maximum a posteriori parameter estimates, automatically
    % estimating the group-level prior.
    %
    % USAGE: results = mfit_optimize(likfun,param,data,[nstarts])
    %
    % INPUTS:
    %   likfun - likelihood function handle
    %   param - [K x 1] parameter structure
    %   data - [S x 1] data structure
    %   nstarts  (optional) - number of random starts    (default: 1)
    %   parallel (optional) - use mfit_optimize_parallel (default: 0)
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
    %               .lme - approximation of the log model evidence (marginal likelihood)
    %               .group - structure containing the group level means (m) and variances (v) of the hyperprior
    %
    % Sam Gershman, July 2017
    
    if nargin < 4; nstarts = 1; end
    if nargin < 5; parallel= 0; end

    
    % initialization
    tol = 1e-3;
    maxiter = 20;
    iter = 0;
    K = length(param);
    S = length(data);
    
    % extract lower and upper bounds
    if ~isfield(param,'lb'); lb = zeros(size(param)) + -inf; else lb = [param.lb]; end
    if ~isfield(param,'ub'); ub = zeros(size(param)) + inf; else ub = [param.ub]; end
    
    % group-level initial parameters
    if all(isinf(lb)) && all(isinf(ub))
        m = randn(1,K);
        v = ones(1,K)*100;
    else
        m = ub + 0.5*(ub-lb);
        v = ub-lb;
    end

    % identity link function is default
    if ~isfield(param, 'link')
        for k = 1:K
            param(k).link = @(x) x;
        end
    end
    
    % run expectation-maximization
    while iter < maxiter
        
        iter = iter + 1;
        disp(['.. iteration ',num2str(iter)]);
        
        % construct prior
        for k = 1:K
            param(k).logpdf = @(x) -0.5 * ((param(k).link(x) - m(k))./sqrt(v(k))).^2 - log((sqrt(2*pi) .* sqrt(v(k))));
        end
        
        % E-step: find individual parameter estimates
        if parallel
            results = mfit_optimize_parallel(likfun,param,data,nstarts);
        else
            results = mfit_optimize(likfun,param,data,nstarts);
        end

        % transform parameters to (-inf, +inf)
        x_orig = results.x;
        for k = 1:K
            results.x(:,k) = param(k).link(results.x(:,k));
        end
        
        % M-step: update group-level parameters
        v = zeros(1,K);
        for s = 1:S
            v = v + results.x(s,:).^2 + diag(pinv(results.H{s}))';
            try
                h = logdet(results.H{s},'chol');
                L(s) = results.logpost(s) + 0.5*(results.K*log(2*pi) - h);
                goodHessian(s) = 1;
            catch
                warning('Hessian is not positive definite');
                try
                    h = logdet(results.H{s});
                    L(s) = results.logpost(s) + 0.5*(results.K*log(2*pi) - h);
                    goodHessian(s) = 0;
                catch
                    warning('could not calculate');
                    goodHessian(s) = -1;
                    L(s) = nan;
                end
            end
        end
        
        m = nanmean(results.x);
        v = max(1e-5, v./S - m.^2);  % make sure variances don't get too small
        L(isnan(L)) = nanmean(L); % interpolate to avoid imaginary numbers
        lme(iter) = sum(L) - K*log(sum([data.N]));
        
        results.group.m = m;
        results.group.v = v;
        results.lme = lme;
        results.goodHessian = goodHessian;        
        results.x = x_orig;
        
        
        if iter > 1 && abs(lme(iter)-lme(iter-1))<tol
            break;
        end
        
    end
