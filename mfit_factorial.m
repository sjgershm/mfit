function [results, bms_results] = mfit_factorial(data,opts,likfun,paramfun)
    
    % Fit a factorial space of models
    %
    % USAGE: [results, bms_results] = mfit_factorial(data,opts,likfun,paramfun)
    %
    % INPUTS:
    %   data - [S x 1] structure array of data for S subjects
    %   opts - [M x 1] structure of model options (see set_opts.m)
    %   likfun - likelihood function handle that takes as input data and
    %            opts (single structures, not arrays)
    %   paramfun - function handle that takes as input opts and returns a
    %           param structure
    %
    % OUTPUTS:
    %   results - [M x 1] model fits
    %   bms_results - Bayesian model selection results
    %
    % Sam Gershman, Jun 2016
    
    for m = 1:length(opts)
        
        disp(['... fitting model ',num2str(m),' out of ',num2str(length(opts))]);
        
        % fit model
        fun = @(x,data) likfun(x,data,opts(m));
        param = paramfun(opts(m));
        results(m) = mfit_optimize(fun,param,data);
        
    end
    
    % Bayesian model selection
    if nargout > 1
        bms_results = mfit_bms(results);
    end