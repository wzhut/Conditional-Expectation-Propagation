rng(0);
rank_list = [3 5 8 10];
batch_size = [500 1000 5000];
n = 5;
% continuous

mse_stat = zeros(4, 5, 3);
rmse_stat = zeros(4, 5, 3);
time = zeros(4, 5, 3);

dim = [200 100 200];
cfg.tol = 1e-5;
cfg.rho = 0.5;
cfg.max_iter = 500;
cfg.verbose = 1;
for nfold = 1 : n
    train = importdata(sprintf('../regression200x100x200/train-fold-%d.txt',nfold),',');
    test = importdata(sprintf('../regression200x100x200/test-fold-%d.txt',nfold),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
        for k = 1 : length(batch_size)
            b = batch_size(k);
            disp(sprintf('nfold: %d rank: %d', nfold, rank));
            tic;
            [mse, rmse, iter, diffs] = c_cpadf(train, test, rank, dim, b, cfg);
            time(r, nfold, k) = toc;
            mse_stat(r, nfold, k) = mse;
            rmse_stat(r, nfold, k) = rmse;
        end
    end
end

save('continuous.mat');

