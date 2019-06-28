function [logp, grad] = gradLogitPosterior(theta, y, x)

N = size(y, 1);
D = size(x, 2);
tx = x * theta; %sum(repmat(theta', [N, 1]) .* x, 2);

% sigma function
sigma = @(x) 1 ./ (1 + exp(-x));

sigma_tx = sigma(tx);

logp = mvnormpdfln(theta) + sum(y .* log(sigma_tx) + (1 - y) .* log(1 - sigma_tx));

grad = -theta + sum(repmat( y - sigma_tx, [1, D]) .* x)';


end

