function opts = mfit_opts(opts)
    
    % Generate default options.
    %
    % USAGE: opts = mfit_opts([opts])
    %
    % INPUTS:
    %   opts - options structure. If empty, all fields will be set to defaults.
    %           .M - number of samples (default: 10000)
    %           .vectorize - is likelihood function vectorized? (default: 1)
    %           .hierarchical - use hierarchical model? (default: 1)
    %           .sd - mean of exponential prior on standard deviations for
    %                 hierarchical model (default: 1)
    %           .nStarts - number of random starts for optimization (default: 2)
    %
    % OUTPUTS:
    %   opts - complete options structure
    %
    % Sam Gershman, June 2015
    
    def_opts.M = 10000;
    def_opts.vectorize = 1;
    def_opts.hierarchical = 1;
    def_opts.sd = 1;
    def_opts.nStarts = 2;
    
    if nargin < 1 || isempty(opts)
        opts = def_opts;
    else
        F = fieldnames(def_opts);
        for f = 1:length(F)
            if ~isfield(opts,F{f})
                opts.(F{f}) = def_opts.(F{f});
            end
        end
    end