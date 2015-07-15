function lik = rllik2(x,data)
    
    % Likelihood function for Q-learning on two-armed bandit with separate
    % learning rates for positive and negative prediction errors.
    %
    % USAGE: lik = rllik2(x,data)
    %
    % INPUTS:
    %   x - parameters:
    %       x(1) - inverse temperature
    %       x(2) - learning rate for positive prediction errors
    %       x(3) - learning rate for negative prediction errors
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
    lr_pos = x(2);
    lr_neg = x(3);
    
    for n = 1:data.N
        c = data.c(n);
        r = data.r(n);
        lik = lik + b*v(c) - logsumexp(b*v,2);
        rpe = r-v(c);
        if rpe <= 0
            v(c) = v(c) + lr_neg*rpe;   % update values (negative prediction error)
        else
            v(c) = v(c) + lr_pos*rpe;   % update values (positive prediction error)
        end
    end