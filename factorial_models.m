function opts = factorial_models(Opts)
    
    % Create a factorial space of models.
    %
    % USAGE: opts = factorial_models(Opts)
    %
    % INPUTS:
    %   Opts - meta options structure, with the same fields as "opts" (see
    %          set_opts.m) but each field accepts a vector of parameters
    %
    % OUTPUTS:
    %   opts - [1 x M] options structure, where each structure in the array
    %          corresponds to one model (a particular combination of settings from Opts)
    %
    % Sam Gershman, Nov 2015
    
    F = fieldnames(Opts);
    for f = 1:length(F)
        X{f} = Opts.(F{f});
    end
    
    g = MyCombVec(X);
    
    for m = 1:size(g,2)
        for f = 1:length(F)
            opts(m).(F{f}) = g(f,m);
            if iscell(opts(m).(F{f})) && length(opts(m).(F{f}))==1; opts(m).(F{f}) = opts(m).(F{f}){1}; end
        end
    end
end

%=========================================================

function out = MyCombVec(X)
    % Generate all possible combinations of input vectors.
    
    if nargin == 0
        out = [];
    else
        out = X{1};
        for i=2:length(X)
            cur = X{i};
            out = [copyb(out,size(cur,2)); copyi(cur,size(out,2))];
        end
    end
end

%=========================================================
function b = copyb(mat,s)
    
    [~,mc] = size(mat);
    inds    = 1:mc;
    inds    = inds(ones(s,1),:).';
    b       = mat(:,inds(:));
    
end

%=========================================================
function b = copyi(mat,s)
    
    [~,mc] = size(mat);
    inds    = 1:mc;
    inds    = inds(ones(s,1),:);
    b       = mat(:,inds(:));
    
end