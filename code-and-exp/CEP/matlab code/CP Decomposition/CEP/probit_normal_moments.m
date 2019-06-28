%mean and variance of prob. \propto \phi(x)\N(x|mu, s)
function [m, v] = probit_normal_moments(mu, s)
    %normalizer
    z = mu./sqrt(1+s);
    r = s./sqrt(1+s);
    N_phi = exp(mvnormpdfln(z')' - normcdfln(z));
    m = mu + r.*N_phi;
    v = s - (r.^2).*N_phi.*(z + N_phi);
end