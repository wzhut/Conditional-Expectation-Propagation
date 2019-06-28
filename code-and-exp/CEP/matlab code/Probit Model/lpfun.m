function [f, g] = lpfun( theta, param )

mu = param.mu;
v = param.v;
y = param.y;
x = param.x;

sigma = @(x) 1 / (1 + exp(-x));

tx = (2 * y - 1)  * x * theta;
cdfln = normcdfln(tx);
f = -cdfln + 0.5 * sum((theta' - mu).^2 ./ v);

if nargout > 1
    g = -exp(mvnormpdfln(tx) - normcdfln(tx)) * (2 * y - 1) * x' + (theta - mu') ./ v';
end

end