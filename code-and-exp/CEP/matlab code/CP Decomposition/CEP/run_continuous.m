rng(0);
rank_list = [3 5 8 10];
n = 5;
% continuous

mse_stat = zeros(4, 5);
rmse_stat = zeros(4 ,5);
time = zeros(4, 5);
dim = [200 100 200];

cep.time_stat = cell(4,5);
cep.rmse_stat = cell(4,5);
cep.iter = zeros(4,5);

cfg.tol = 1e-4;
cfg.rho = 0.5;
cfg.max_iter = 500;
cfg.verbose = 1;
for nfold = 1 : n
    train = importdata(sprintf('../regression200x100x200/train-fold-%d.txt',nfold),',');
    test = importdata(sprintf('../regression200x100x200/test-fold-%d.txt',nfold),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
        disp(sprintf('nfold: %d rank: %d', nfold, rank)); 
%         tic;
        [mse, rmse, iter, diffs, test_rmses, time] = c_cpcep(train, test, rank, dim, cfg);
%         time(r, nfold) = toc;
        cep.rmse_stat{r, nfold} = sqrt(test_rmses);
        cep.time_stat{r, nfold} = cumsum(time);
        cep.iter(r, nfold) = iter;
        mse_stat(r, nfold) = mse;
        rmse_stat(r, nfold) = rmse;
    end
end

save('cep-continuous.mat', 'cep');

