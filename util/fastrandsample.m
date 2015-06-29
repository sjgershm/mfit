function y = fastrandsample(p,n)
    
    % Multinomial random numbers.
    %
    % USAGE: y = fastrandsample(p,n)
    
    if nargin < 2; n=1; end
    [~, y] = histc(rand(1,n),[0 cumsum(p)]);