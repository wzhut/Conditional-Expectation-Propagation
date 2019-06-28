rng(0);

num_sample = 10000;
D = 4;

% sample theta
theta = mvnrnd(zeros(D,1), eye(D));

% sample x
x = mvnrnd(zeros(D,1), eye(D), num_sample);

% label y
y = binornd(ones([num_sample 1]), normcdf(sum(repmat(theta, [num_sample 1]) .* x, 2)));

num_train = ceil(num_sample / 2);

% train 
train.x = x(1:num_train, :);
train.y = y(1:num_train, :);
% test
test.x = x(num_train + 1:end, :);
test.y = y(num_train + 1:end, :);

% estimate mean and variance

theta0 = zeros(D,1);
        
f = @(theta) gradProbitPosterior(theta, train.y, train.x);
%burning in
n_warmup = 100000;
%after burn-in
n_mcmc_samples = 50000;
[samples, logp_samples] = NUTS_wrapper(f, theta0, n_warmup, n_mcmc_samples);
%         plot(1:n_mcmc_samples, logp_samples);
res_samples = samples(:, 1:10:end);
%         ess = ESS(samples);

sample_size = size(res_samples, 2);
ts_mean = mean(res_samples,2);
ts_var = zeros(D,D);
for k = 1 : sample_size
    ts_var = ts_var + (res_samples(:, k) - ts_mean) * (res_samples(:, k) - ts_mean)';
end
ts_var = ts_var / (sample_size - 1);

save('simulation1', 'train', 'test', 'ts_mean', 'ts_var');



