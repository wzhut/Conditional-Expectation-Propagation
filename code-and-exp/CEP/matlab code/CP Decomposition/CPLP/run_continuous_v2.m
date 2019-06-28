clear;
addpath(genpath('../minFunc_2012'));
rng(0);
k = 5;
rank_list = [3 5 8 10];

lp.time_stat = cell(4,5);
lp.rmse_stat = cell(4,5);
lp.iter = zeros(4,5);

cfg.tol = 1e-4;
cfg.rho = 0.5;
cfg.max_iter = 500;
cfg.verbose = 1;

for i=1:k
    train = importdata(sprintf('../regression200x100x200/train-fold-%d.txt',i),',');
    test = importdata(sprintf('../regression200x100x200/test-fold-%d.txt',i),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
        [~, ~, ~, iter, test_rmses, time] = c_cplp_v2(train, test, rank, cfg);

        cep.rmse_stat{r, nfold} = test_rmses;
        cep.time_stat{r, nfold} = cumsum(time);
        cep.iter(r, nfold) = iter;

    end
end

save('lp-continuous-performance.mat', 'lp');


