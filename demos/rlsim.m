function data = rlsim(x,R,N)
    
    % Simulate data from reinforcement learning agent on a two-armed bandit.
    %
    % USAGE: data = rlsim(x,R,N)
    %
    % INPUTS:
    %   x - parameter vector:
    %       x(1) - inverse temperature
    %       x(2) - learning rate
    %   R - [1 x 2] reward probabilities for each arm
    %   N - number of trials
    %
    % OUTPUTS:
    %   data - structure with the following fields
    %           .c - [N x 1] choices
    %           .r - [N x 1] rewards
    %
    % Sam Gershman, June 2015
    
    v = [0 0];  % initial values
    data.N = N;
    b = x(1);
    lr = x(2);
    
    for n = 1:N
        p = exp(b*v - logsumexp(b*v,2));  % softmax choice probability
        c = fastrandsample(p);            % random choice
        r = double(rand<R(c));            % reward feedback
        v(c) = v(c) + lr*(r-v(c));        % update values
        data.c(n,1) = c;
        data.r(n,1) = r;
    end