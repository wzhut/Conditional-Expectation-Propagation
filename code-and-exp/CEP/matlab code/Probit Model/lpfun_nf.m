function [f, g] = lpfun_nf( theta, param )

mu = param.mu;
precision = param.precision;
y = param.y;
x = param.x;

sigma = @(x) 1 / (1 + exp(-x));

tx = (2 * y - 1)  * x * theta;
cdfln = normcdfln(tx);
f = -cdfln + 0.5 * (theta - mu)' * precision * (theta - mu);

if nargout > 1
    g = -normpdf(tx) / normcdf(tx) * (2 * y - 1) * x' + precision * (theta - mu);
end

end