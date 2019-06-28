mu = 10;
sigma = 4;

[nd,weights] = quadrl(9);

x = nd * sigma + mu;

mean_x = x * weights;
var_x = x.^2 * weights - mean_x^2;
