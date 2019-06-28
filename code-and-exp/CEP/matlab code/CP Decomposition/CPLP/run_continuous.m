clear;
addpath(genpath('../minFunc_2012'));
rng(0);
k = 5;
rank_list = [3 5 8 10];
% mse_stat = zeros(4, 5);
% rmse_stat = zeros(4, 5);
% time = zeros(4, 5);

lp.time_stat = cell(4,5);
lp.rmse_stat = cell(4,5);
lp.iter = zeros(4,5);

cfg.tol = 0;
cfg.rho = 0.5;
cfg.max_iter = 5;
cfg.verbose = 1;
for i=1:k
    train = importdata(sprintf('../regression200x100x200/train-fold-%d.txt',i),',');
    test = importdata(sprintf('../regression200x100x200/test-fold-%d.txt',i),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
%         tic;
        [mse, rmse, iter, diffs, test_rmses, time] = c_cplp(train, test, rank, [200 100 200], cfg);
%         time(r, i) = toc;
        lp.rmse_stat{r, i} = sqrt(test_rmses);
        lp.time_stat{r, i} = cumsum(time);
        lp.iter(r, i) = iter;
        
%         mse_stat(r, i) = mse;
%         rmse_stat(r, i) = rmse;
    end
end

save('lp-continuous-5.mat', 'lp');


