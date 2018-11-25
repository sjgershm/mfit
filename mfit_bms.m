function bms_results = mfit_bms(results,use_bic)
    
    % Bayesian model selection for group studies. Uses the Laplace
    % approximation to the marginal likelihood. If the Hessian is
    % degenerate, it resorts to the Bayesian information criterion.
    % See bms.m for more information.
    %
    % USAGE: bms_results = mfit_bms(results,[use_bic])
    %
    % INPUTS:
    %   results - [J x 1] results structure, where J is the number of models
    %   use_bic (optional) - use BIC instead of Laplace approximation? (default: 0)
    %
    % OUTPUTS:
    %   bms_results - structure with the following fields:
    %       .alpha   - vector of model probabilities
    %       .exp_r   - expectation of the posterior p(r|y)
    %       .xp      - exceedance probabilities
    %       .pxp     - protected exceedance probabilities
    %       .bor     - Bayes Omnibus Risk (probability that model frequencies are equal)
    %       .g       - posterior belief g(i,k)=q(m_i=k|y_i) that model k generated the data for the i-th subject
    %
    % REFERENCES:
    %   Stephan KE, Penny WD, Daunizeau J, Moran RJ, Friston KJ (2009)
    %   Bayesian Model Selection for Group Studies. NeuroImage 46:1004-1017
    %
    %   Rigoux, L, Stephan, KE, Friston, KJ and Daunizeau, J. (2014)
    %   Bayesian model selection for group studies—Revisited.
    %   NeuroImage 84:971-85. doi: 10.1016/j.neuroimage.2013.08.065
    %
    % Sam Gershman, June 2015
    
    if nargin < 2; use_bic = 0; end
    
    for j = 1:length(results)
        lme0(:,j) = -0.5*results(j).bic;
        for s = 1:length(results(j).H); h(s,1) = log(det(results(j).H{s})); end
        lme(:,j) = results(j).logpost' + 0.5*(results(j).K*log(2*pi) - h);
    end
    
    ix = isnan(lme)|isinf(lme)|imag(lme)~=0; % use BIC if any Hessians are degenerate
    if any(ix(:))
        lme = lme0;
    end
    if use_bic==1; lme = lme0; end

    lme(any(isnan(lme)|isinf(lme),2),:) = [];
    
    [bms_results.alpha, bms_results.exp_r, bms_results.xp, bms_results.pxp, bms_results.bor, bms_results.g] = bms(lme);