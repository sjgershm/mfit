function [logp, results] = mfit_crossval(likfun,param,folds,nstarts)
    
    % Cross-validation.
    %
    % USAGE: [logp, results] = mfit_crossval(likfun,param,folds,[nstarts])
    %
    % INPUTS:
    %   likfun - likelihood function handle
    %   param - [K x 1] parameter structure
    %   folds - [F x 1] data structure for F folds, where each fold
    %           contains a 'data' and 'testdata' field. Model is fit to data and
    %           tested on testdata.
    %   nstarts (optional) - number of random starts (default: 5)
    %
    % OUTPUTS:
    %   logp - [S x F] log predictive probabilities for each subject and fold
    %   results - results structure from optimization
    %
    % Sam Gershman, Sep 2015
    
    if nargin < 4; nstarts = []; end
    
    for i = 1:length(folds)
        disp(['... fold ',num2str(i)]);
        results(i) = mfit_optimize(likfun,param,folds(i).data,nstarts);
        logp(:,i) = mfit_predict(folds(i).testdata,results(i));
    end