function [ xn ] = c_LBFGS( param )

m = 10;
v = param.v;

dim = size(v, 1);
rank = size(v{1}, 1);

iter = 1;
H0 = eye(dim * rank + 1);
x_list = zeros(dim * rank + 1, m - 1);
g_list = zeros(dim * rank + 1, m - 1);

diff = 1;
H = H0;
current = 1;
x_list
while diff > 0.001
    g = c_gradient(param);
    % save g
    g_list(current) = g;
    
    d = - H * g;
    alpha = 1;
    t = 0.9;
    x0 = x_list(current);
    while False
        alpha = t * alpha;
    end
    
    x1 = x0 + alpha * d;
    x0 = x1;
    % next
    current = mod(current + 1, m) + 1;
    % save x
    x_list(current) = x0;
    
end



end

