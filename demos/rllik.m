function lik = rllik(x,data)
    
    % Likelihood function for Q-learning on two-armed bandit with single
    % learning rate.
    %
    % USAGE: lik = rllik(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - learning rate
    %   data - structure with the following fields
    %          .c - [N x 1] choices
    %          .r - [N x 1] rewards
    %
    % OUTPUTS:
    %   lik - log-likelihood
    %
    % Sam Gershman, June 2015
    
    C = max(unique(data.c)); % number of options
    v = zeros(1,C);  % initial values
    lik = 0;
    b = x(1);
    lr = x(2);
    
    for n = 1:data.N
        c = data.c(n);
        r = data.r(n);
        lik = lik + b*v(c) - logsumexp(b*v,2);
        rpe = r-v(c);
        v(c) = v(c) + lr*rpe;      % update values
    end