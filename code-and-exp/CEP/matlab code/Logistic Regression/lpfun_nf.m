function [f, g] = lpfun_nf( theta, param )

mu = param.mu;
precision = param.precision;
y = param.y;
x = param.x;

sigma = @(x) 1 / (1 + exp(-x));

tx = x * theta;
sigma_tx = sigma(tx);
f = -y * log(sigma_tx) - (1 - y) * log(1 - sigma_tx) + 0.5 * (theta - mu)' * precision * (theta - mu);

if nargout > 1
    g = -(y - sigma_tx) * x' + precision * (theta - mu);
end

end