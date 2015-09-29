function param = mfit_priorfit(x,param)
    
    % Fit prior hyperparameters.
    %
    % USAGE: param = mfit_priorfit(x,param)
    %
    % INPUTS:
    %   x - [S x K] parameter estimates for S subjects and K parameters
    %   param - parameter structure with a 'fit' field that specifies the
    %           hyperparameter estimation function, taking as input a vector of
    %           parameter estimates and returning a vector of hyperparameter estimates 
    %
    % OUTPUTS:
    %   param - same as input, with 'hp' field storing hyperparameter estimates
    %
    % Sam Gershman, Sep 2015
    
    for i = 1:length(param)
        x(:,i) = min(max(x(:,i),param(i).lb),param(i).ub);
        param(i).hp = param(i).fit(x(:,i));
    end