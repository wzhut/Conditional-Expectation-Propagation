addpath(genpath('../lightspeed'));
addpath(genpath('../NUTS'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

for i = 1 : num_dataset
    for nfold = 1 : num_run
        train_data = importdata(sprintf('./%s/train-%d.csv', data_set{i}, nfold));
        test_data = importdata(sprintf('./%s/test-%d.csv', data_set{i}, nfold));
        
        train.x = train_data(:, 1: end-1);
        train.y = train_data(:, end);
        test.x = test_data(:, 1:end-1);
        test.y = test_data(:, end);
        D = size(train.x, 2);
%         theta0 = zeros(D,1);
%         
%         f = @(theta) gradProbitPosterior(theta, train.y, train.x);
%         %burning in
%         n_warmup = 100000;
%         %after burn-in
%         n_mcmc_samples = 50000;
%         [samples, logp_samples] = NUTS_wrapper(f, theta0, n_warmup, n_mcmc_samples);
% %         plot(1:n_mcmc_samples, logp_samples);
%         res_samples = samples(:, 1:10:end);
% %         ess = ESS(samples);
%         
%         sample_size = size(res_samples, 2);
%         ts_mean = mean(res_samples,2);
%         ts_var = zeros(D,D);
%         for k = 1 : sample_size
%             ts_var = ts_var + (res_samples(:, k) - ts_mean) * (res_samples(:, k) - ts_mean)';
%         end
%         ts_var = ts_var / (sample_size - 1);
          ts_mean = zeros(D,1);
          ts_var = eye(D);
        
        save(sprintf('./%s/nfold-%d.mat',data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');
    end
end