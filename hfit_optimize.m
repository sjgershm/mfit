function results = hfit_optimize(likfun,hyparam,param,data,nstarts,verbose,opts)

    % Find MAP hyperparameters and MAP parameters conditioned on them.
    %
    % Generative model:
    %   group-level hyperparameters h ~ P(h), defined by hyparam.logpdf
    %   individual parameters       x ~ P(x|h), defined by param.hlogpdf
    %   data                        D ~ P(D|x), defined by likfun 
    %
    % Uses expectation maximization (EM) to find MAP P(h|D), with x serving as latent
    % variables (Bishop 2006). Approximates the integral over x using importance
    % sampling, with x's resampled every few iterations using Metropolis-within-Gibbs
    % sampling. Then finds MAP P(x|h,D) using mfit_optimize.
    %
    % USAGE: results = hfit_optimize(likfun,hyparam,param,data,verbose)
    %
    % INPUTS:
    %   likfun - likelihood function handle
    %   hyparam - [K x 1] hyperparameter structure
    %   param - [K x 1] parameter structure, with hlogpdf and hrnd fields
    %   data - [S x 1] data structure
    %   verbose (optional) - whether to print stuff
    %
    % OUTPUTS:
    %   results - results of mfit_optimize, plus:
    %           .h - MAP hyperparameters h
    %           .param - the param structure array with .logpdf defined based on h
    %
    % Momchil Tomov, Aug 2018

    if nargin < 5
        nstarts = 5;
    end
    if nargin < 6
        verbose = false;
    end
    if nargin < 7
        opts = hfit_default_opts
    end

    % fit hyperparameters h
    disp('\n\n ------------- fitting hyperparameters -----------\n\n');
    tic
    h = EM(likfun, hyparam, param, data, opts, verbose);
    toc

    % set .logpdf according to h
    param = set_logpdf(hyparam, param, h); 

    % fit parameters x given h
    disp('\n\n ------------- fitting parameters -----------\n\n');
    tic
    results = mfit_optimize(likfun, param, data, nstarts);
    toc

    % correct for hyperparameters
    hyK = 0;
    for k = 1:length(hyparam)
        hyK = hyK + length(hyparam(k).lb);
    end
    for s = 1:length(data)
        results.bic(s,1) = results.bic(s,1) + hyK * log(data(s).N);
        results.aic(s,1) = results.aic(s,1) + hyK * 2;
    end

    results.h = h;
    results.param = param; % with .logpdf set
end


% Expectation maximization to find MAP P(h|D)
%
function h = EM(likfun, hyparam, param, data, opts, verbose);
    % initialization
    tol = opts.tol; % tolerance: stop when improvements in Q are less than that
    maxiter = opts.maxiter; % stop after that many iterations
    nsamples = opts.nsamples; % how many samples of x to use to approximate EM integral
    batch_size = opts.batch_size; % subsample once every batch_size samples (gibbs sampler param)
    burn_in = opts.burn_in; % burn in first few samples when gibbs sampling
    init_samples = opts.init_samples; % how many samples of h to use to initialize h_old
    lme_samples = opts.lme_samples; % how many samples to use to approximate P(D|h) (for display purposes only)
    eff_min = opts.eff_min; % minimum effective sample size (ratio) to trigger resampling

    assert(length(hyparam) == length(param), 'param and hyparam must have the same length');

    % pick a crude estimate of h_old 
    % compute a bunch of random h's and pick one with max P(h|D)
    disp('Initializing h_old with a crude estimate ...');
    [h_old, logmargpost_old] = random_MAP(likfun, hyparam, param, data, init_samples, lme_samples, verbose);

    disp(['h_old initialized as ', mat2str(h_old)]);
    disp(['ln P(h|D) ~= ', num2str(logmargpost_old)]);

    % Draw samples from q(x) = P(x|D,h_old)
    [X, logq] = sample(h_old, likfun, hyparam, param, data, nsamples, batch_size, burn_in, verbose);

    options = optimset('Display','off');
    warning off all
    
    % run expectation-maximization
    iter = 0;
    while iter < maxiter
        
        iter = iter + 1;
        disp(['.. iteration ',num2str(iter)]);

        %
        % E step: use importance sampling to approximate 
        % the integral over P(x|D,h_old)
        %

        % recompute importance weights and effective sample size
        [w, eff] = weights(X, data, h_old, hyparam, param, likfun, logq);

        disp(['  E step: effective sample size = ', num2str(eff), ' (out of ', num2str(length(w)), ')']);

        % redraw new samples from q(x) = P(x|D,h_old) if effective sample size is too small
        if eff < length(w) * eff_min
            [X, logq] = sample(h_old, likfun, hyparam, param, data, nsamples, batch_size, burn_in, verbose);
            [w, eff] = weights(X, data, h_old, hyparam, param, likfun, logq);
            disp(['           new effective sample size = ', num2str(eff), ' (out of ', num2str(length(w)), ')']);
        end

        %
        % M step: maximize Q(h,h_old)
        %
        disp('  M step: maximizing Q...');

        f = @(h_new) -computeQ(h_new, h_old, X, w, data, hyparam, param, likfun, verbose);

        %h0 = hyparam_rnd(hyparam, param); TODO try and compare
        h0 = h_old;

        [h_new,nQ] = fmincon(f,h0,[],[],[],[],[hyparam.lb],[hyparam.ub],[],options);

        % bookkeeping
        Q_old = -f(h_old);
        Q_new = -nQ;
        logmargpost_new = loghypost(h_new, data, hyparam, param, likfun, lme_samples, verbose);

        % print stuff
        disp(['     old Q = ', num2str(Q_old)]);
        disp(['     new Q = ', num2str(Q_new)]);
        if verbose
            disp(['ln weights = ', mat2str(w)]);
        end
        disp(['    h_new = ', mat2str(h_new)]);
        disp(['    ln P(h_new|D) ~= ', num2str(logmargpost_new)]);
        if verbose
            disp('        ...vs...');
            disp(['    h_old = ', mat2str(h_old)]);
            disp(['    ln P(h_old|D) ~= ', num2str(logmargpost_old), ' (new one should be better)']);
        end

        % termination condition
        if iter > 1 && abs(Q_new - Q_old) < tol
            break;
        end

        h_old = h_new;
        logmargpost_old = logmargpost_new;
    end

    h = h_new;
end


% find h = argmax P(h|D) for random samples of h
%
function [h_best, logp_best] = random_MAP(likfun, hyparam, param, data, init_samples, lme_samples, verbose);
    logp_best = -Inf;
    h_best = [];
    for i = 1:init_samples
        if verbose, disp(['  init iter ', num2str(i)]); end
        h = hyparam_rnd(hyparam, param);
        logp = loghypost(h, data, hyparam, param, likfun, lme_samples, verbose);
        if logp > logp_best
            h_best = h;
            logp_best = logp;
            if verbose
                disp(['    new h_best = ', mat2str(h_best)]);
                disp(['    ln P(h|D) = ', num2str(logp_best)]);
            end
        end
    end
end


% compute importance weights w for a given set of parameter samples
%
function [w, eff] = weights(X, data, h_old, hyparam, param, likfun, logq)
    % compute ln P(x|D,h_old) for samples (unnormalized)
    logp = logpost(X, data, h_old, hyparam, param, likfun);

    % compute importance weights
    % w(i) = p(i)/q(i) / sum(p(j)/q(j))   (Bishop 2006, p. 533)
    logw = logp - logq - logsumexp(logp - logq);
    w = exp(logw);

    % compute effective sample size (Gelman 2013 BDA, p. 266)
    eff = 1 / exp(logsumexp(logw * 2));
end


% Draw random samples of paramters from P(x|D,h) using Gibbs sampling (Bishop 2006, p. 543).
%
% OUTPUTS:
%   X = [S x K x nsamples] samples; each row is a set of parameters x
%   logq = [1 x nsamples] = ln q(x) = ln P(x|D,h) for each x (unnormalized); used to compute importance weights

%
function [X, logq] = sample(h_old, likfun, hyparam, param, data, nsamples, batch_size, burn_in, verbose)
    K = length(param);
    S = length(data);
    X = nan(S, K, nsamples);

    % initialize x0
    x_old = param_rnd(hyparam, param, h_old, data, true);

    % set .logpdf = P(x|h) for convenience
    param = set_logpdf(hyparam, param, h_old);

    % log of std of proposal distribution for each component
    ls = zeros(S,K);

    disp('    Drawing new samples from P(x|D,h)...');

    % Draw nsamples samples
    for n = 1:nsamples + burn_in 
        x_new = nan(S,K);

        if verbose, disp(['     sample ', num2str(n)]); end

        % Adaptive Metropolis-within-Gibbs for each component
        % (Roberts and Rosenthal, 2008)
        for s = 1:S
            for k = 1:K
                % P(x_k|x_\k,D,h) is proportional to P(D|x) P(x_k|h)
                logpost_s_k = @(x_s_k) likfun([x_new(s,1:k-1) x_s_k x_old(s,k+1:end)], data(s)) + param(k).logpdf(x_s_k);

                % proposals update by adding an increment ~ N(0,exp(ls(s,k))
                proprnd = @(x_s_k_old) mh_proprnd(x_s_k_old, param, ls(s,k), k);

                % draw batch_size samples of (s,k)-th component and take the last one only
                [x_s_k, accept] = mhsample(x_old(s,k), batch_size, 'logpdf', logpost_s_k, 'proprnd', proprnd, 'symmetric', true);
                x_new(s,k) = x_s_k(batch_size); % ignore first batch_size-1 samples

                % update proposal distribution to maintain the acceptance rate
                % around 0.44
                d = min(0.01, sqrt(n));
                if accept > 0.44
                    ls(s,k) = ls(s,k) + d;
                else
                    ls(s,k) = ls(s,k) - d;
                end
            end
        end

        X(:,:,n) = x_new;
        x_old = x_new;
    end

    % discard burn-in samples
    X = X(:,:,burn_in+1:end);

    % compute q(x) = P(x|D,h) up to a proportionality constant
    logq = logpost(X, data, h_old, hyparam, param, likfun);
end


% Metropolis proposal function for x(s,k)_new given x(s,k)_old
% update by adding an increment ~ N(0,exp(ls(s,k))
%
function x_s_k_new = mh_proprnd(x_s_k_old, param, ls_s_k, k)
    while true
        x_s_k_new = normrnd(x_s_k_old, exp(ls_s_k));
        if param(k).lb <= x_s_k_new && x_s_k_new <= param(k).ub
            % keep parameters within bounds
            break;
        end
    end
end


% ln P(x|h) = sum ln P(x(s)|h)
%
function logp = logprior(x, h, hyparam, param)
    logp = 0;
    for s = 1:size(x,1)
        i = 1;
        for k = 1:length(param)
            l = length(hyparam(k).lb);
            logp = logp + param(k).hlogpdf(x(s,k),h(i:i+l-1));
            i = i + l;
        end
    end
    %disp(['  logprior ', mat2str(x), ' = ', num2str(logp)]);
end


% ln P(D|x) = sum ln P(D(s)|x(s))
%
function logp = loglik(x, data, likfun)
    assert(size(x,1) == length(data));
    logp = 0;
    for s = 1:length(data)
        logp = logp + likfun(x(s,:),data(s));
    end
    %disp(['  loglik ', mat2str(x), ' = ', num2str(logp)]);
end


% ln P(x|D,h) = ln P(D|x) + ln P(x|h) + const
% for all samples x in X, up to a proportionality constant.
% X = [S x K x nSamples]
%
function logp = logpost(X, data, h, hyparam, param, likfun)
    nsamples = size(X,3);
    for n = 1:nsamples
        x = X(:,:,n);
        logp(n) = loglik(x, data, likfun) + logprior(x, h, hyparam, param);
    end
end


% ln P(h)
%
function logp = loghyprior(h, hyparam)
    logp = 0;
    i = 1;
    for k = 1:length(hyparam)
        l = length(hyparam(k).lb);
        logp = logp + hyparam(k).logpdf(h(i:i+l-1));
        i = i + l;
    end
end
 

% ln P(D|h) crude approximation (log model evidence)
%
% P(D|h) = integral P(D|x) P(x|h) dx
%       ~= 1/L sum P(D|x^l)
% where the L samples x^1..x^l are drawn from P(x|h)
%
function logp = loghylik(h, data, hyparam, param, likfun, nsamples, verbose)
    logp = [];
    for n = 1:nsamples
        x = param_rnd(hyparam, param, h, data, true, 5);
        lik = loglik(x, data, likfun);
        if ~isnan(lik) && ~isinf(lik)
            logp = [logp; lik];
        end
    end
    %disp(['LME computation P(D|h) using ', num2str(length(logp)), ' samples']);
    if isempty(logp)
        if verbose, disp(['     No good parameters x were found for the given h= ', mat2str(h)]); end
        logp = -Inf;
    else
        logp = logsumexp(logp) - log(length(logp));
    end
end


% ln P(h|D) crude approximation (up to a proportionality constant)
%
function logp = loghypost(h, data, hyparam, param, likfun, nsamples, verbose)
    logp = loghylik(h, data, hyparam, param, likfun, nsamples, verbose) + loghyprior(h, hyparam);
end


% set .logpdf i.e. P(x|h) for each param based on given hyperparameters h
%
function param = set_logpdf(hyparam, param, h)
    i = 1;
    for k = 1:length(param)
        l = length(hyparam(k).lb);
        param(k).logpdf = @(x) param(k).hlogpdf(x,h(i:i+l-1));
        i = i + l;
    end
end


% random draw from P(h)
%
function h = hyparam_rnd(hyparam, param)
    h = [];
    for k = 1:length(hyparam)
        while true
            h_k = hyparam(k).rnd();
            x_k = param(k).hrnd(h_k);
            logp = param(k).hlogpdf(x_k, h_k);
            if all(hyparam(k).lb <= h_k) && all(h_k <= hyparam(k).ub) && ~isinf(logp) && ~isnan(logp)
                % keep hyperparameters within bounds and do a sanity check
                % to make sure they don't generate invalid parameters
                h = [h h_k];
                break;
            end
        end
    end
end


% random draw from P(x|h)
%
function x = param_rnd(hyparam, param, h, data, respect_bounds, max_attempts)
    if nargin < 6
        max_attempts = 1000;
    end
    for s = 1:length(data)
        i = 1;
        for k = 1:length(param)
            l = length(hyparam(k).lb);
            for attempts = 1:max_attempts
                x(s,k) = param(k).hrnd(h(i:i+l-1));
                if ~respect_bounds || (param(k).lb <= x(s,k) && x(s,k) <= param(k).ub)
                    % keep parameters within bounds
                    break;
                elseif attempts == max_attempts
                    %disp(['  Could not find good setting for param ', num2str(k), ' after ', num2str(attempts), ' attempts with h = ', mat2str(h)]);
                    x(s,k) = NaN;
                end
            end
            i = i + l;
        end
    end
end



% Q(h|h_old) = ln P(h) + integral P(x|D,h_old) ln P(D,x|h) dx
%            = ln P(h) + integral P(x|D,h_old) [ln P(D|x) + ln P(x|h)] dx
%           ~= ln P(h) + sum w(l) [ln P(D|x^l) + ln P(x^l|h)]
% (Bishop 2006, p. 441 and p. 536)
% Last line is an importance sampling approximation of the integral
% using L samples x^1..x^L, with the importance weight of sample l given by
% w(l) = P(x^l|data,h_old) / q(x^l) / sum(...)      (Bishop 2006, p. 533)
%
% TODO optimization: can ignore P(D|x) when maximizing Q w.r.t. h
%
function Q = computeQ(h_new, h_old, X, w, data, hyparam, param, likfun, verbose)
    nsamples = size(X,3);

    % integral P(x|D,h_old) ln P(D,x|h) dx
    % approximated using importance sampling
    Q = 0;
    for n = 1:nsamples
        x = X(:,:,n);
        Q = Q + w(n) * (loglik(x, data, likfun) + logprior(x, h_new, hyparam, param));
    end 

    % ln P(h)
    Q = Q + loghyprior(h_new, hyparam); 

    if verbose
        disp(['   computeQ: h_new = ', mat2str(h_new)]);
        disp(['       P(h) = ', num2str(loghyprior(h_new, hyparam)), ' Q = ', num2str(Q)]);
    end
end
