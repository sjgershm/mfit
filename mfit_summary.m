function mfit_summary(results,bootfun,nboot)
    
    % Summarize parameter estimates (bootstrapped 95% confidence
    % intervals).
    %
    % USAGE: mfit_summary(results,[bootfun],[nboot])
    %
    % INPUTS:
    %   results - results structure
    %   bootfun (optional) - bootstrap function (default: @mean)
    %   nboot (optional) - number of bootstrap samples (default: 1000)
    %
    % Sam Gershman, March 2017
    
    if nargin < 3 || isempty(bootfun); bootfun = @mean; end
    if nargin < 4 || isempty(nboot); nboot = 1000; end
    
    for i = 1:length(results.param)
        ci = bootci(nboot,bootfun,results.x(:,i));
        disp([results.param(i).name,': [',num2str(ci(1)),', ',num2str(ci(2)),']']);
    end