function [f, g] = lpfun( theta, param )

mu = param.mu;
v = param.v;
y = param.y;
x = param.x;

sigma = @(x) 1 / (1 + exp(-x));

tx = x * theta;
sigma_tx = sigma(tx);
f = -y * log(sigma_tx) - (1 - y) * log(1 - sigma_tx) + 0.5 * sum((theta' - mu).^2 ./ v);

if nargout > 1
    g = -(y - sigma_tx) * x' + (theta - mu') ./ v';
end

end