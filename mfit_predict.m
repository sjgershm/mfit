function logp = mfit_predict(data,results)
    
    % Compute log predictive probability of new data.
    %
    % USAGE: logp = mfit_predict(data,results)
    %
    % INPUTS:
    %   data - [S x 1] data structure
    %   results - results structure
    %
    % OUTPUTS:
    %   logp - [S x 1] log predictive probabilities for each subject
    %
    % Sam Gershman, June 2015
    
    S = length(data);
    logp = zeros(S,1);
    
    for s = 1:S
        logp(s) = results.likfun(results.x(s,:),data(s));
    end