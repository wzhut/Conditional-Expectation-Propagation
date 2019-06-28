function [logp, grad] = gradProbitPosterior(theta, y, x)

N = size(y, 1);
D = size(x, 2);
tmp = (2 * y - 1) .* sum(repmat(theta', [N, 1]) .* x, 2);

pdf = mvnormpdf(tmp');
cdf = normcdf(tmp');

logp = mvnormpdfln(theta) + sum(normcdfln(tmp));
grad = -theta;

grad = grad + sum(repmat(pdf ./ cdf .* (2 * y' - 1), [D, 1]) .* x', 2);


end

