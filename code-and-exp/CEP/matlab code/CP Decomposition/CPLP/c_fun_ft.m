function [f, g] = c_fun( x, param ) %y, v, mu, precision, tau, alpha, beta )

y = param.y;
mu = param.mu;
precision = param.precision;
alpha = param.alpha;
beta = param.beta;
rank = param.r;

tau = alpha / beta;

dim = size(x, 1) / rank; 

v = cell(dim, 1);

for i = 1: dim
    v{i} = x((i - 1) * rank + 1: i * rank);
end

shared = ones(rank,1);

for i = 1 : dim
    shared = shared .* v{i};
end
shared = sum(shared) - y;

% function value
f =  0.5 * shared^2 * tau;

for i = 1 : dim
    f = f + 0.5 * (v{i} - mu{i})' * precision{i} * (v{i} - mu{i}); 
end


if nargout > 1
    g = zeros(dim * rank, 1);
    % gradient
    for i = 1 : dim
        select = setdiff(1:dim, i);
        tmp = ones(rank,1);
        for k = 1 : dim - 1
            tmp = tmp .* v{select(k)};
        end
        g((i-1)*rank + 1:i*rank) = tau * shared * tmp + precision{i} * (v{i} - mu{i});
    end
end


end

