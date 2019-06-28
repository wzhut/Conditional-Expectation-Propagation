function [ f, g ] = b_fun( x, param ) %y, v, mu, pre)

y = param.y;
mu = param.mu;
precision = param.precision;
rank = param.r;

dim = size(x, 1) / rank; 
v = cell(dim, 1);

for i = 1: dim
    v{i} = x((i - 1) * rank + 1: i * rank);
end

shared = ones(rank,1);
for i = 1 : dim
    shared = shared .* v{i};
end
shared = (2 * y - 1)*sum(shared);


% function value
f = -log(normcdf(shared) + realmin);
for i = 1 : dim
    f = f + 0.5 * (v{i} - mu{i})' * precision{i} * (v{i} - mu{i});
end

shared = -normpdf(shared) / (normcdf(shared) + realmin) * (2 * y - 1);


if nargout > 1
    g = zeros(dim * rank, 1);
    for i = 1 : dim
        select = setdiff(1:dim, i);
        tmp = ones(rank, 1);
        for k = 1 : dim - 1
            tmp = tmp .* v{select(k)};
        end
        g((i - 1) * rank + 1:i * rank) = shared * tmp + precision{i} * (v{i} - mu{i});
    end
end


end

