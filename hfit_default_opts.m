function opts = hfit_default_opts()

% Default set of options to use in hfit_optimize.
% Can initialize opts with this and then modify fields accordingly.
%
% USAGE:
%   opts = hfit_default_opts
%
% OUTPUT:
%   opts - structure with the following fields:
%          opts.tol - tolerance: stop when improvements in Q are less than that
%          opts.maxiter - stop after that many iterations
%          opts.nsamples - how many samples of x to use to approximate EM integral
%          opts.batch_size - subsample once every batch_size samples in gibbs sampler
%          opts.burn_in - burn in first few samples when gibbs sampling
%          opts.init_samples - how many samples of h to use to initialize h_old
%          opts.lme_samples - how many samples to use to approximate P(D|h) (for display purposes only)
%          opts.eff_min - minimum effective sample size (ratio) to trigger resampling
%

opts.tol = 1e0;
opts.maxiter = 20;
opts.nsamples = 100;
opts.batch_size = 50;
opts.burn_in = 50; 
opts.init_samples = 100;
opts.lme_samples = 10;
opts.eff_min = 0.25;

end
