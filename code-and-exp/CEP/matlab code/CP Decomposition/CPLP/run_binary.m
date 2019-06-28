clear;
addpath(genpath('../minFunc_2012'));
rng(0);
k = 5;
rank_list = [3 5 8 10];
auc_stat = zeros(4,5);
time = zeros(4,5);

lp.test_auc_stat = cell(4, 5);
lp.time_stat = cell(4,5);
lp.iter = zeros(4,5);

is_binary = true;

cfg.tol = 1e-4;
cfg.rho = 0.1;
cfg.max_iter = 5;
cfg.verbose = 1;
for i=1:k
    train = importdata(sprintf('../complemented_data/train-fold-%d.txt',i),',');
    test = importdata(sprintf('../complemented_data/test-fold-%d.txt',i),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
        [auc, iter, test_aucs, time] = b_cplp(train, test, rank, [203 203 200],cfg);
        lp.test_auc_stat{r, i} = test_aucs;
        lp.time_stat{r, i} = cumsum(time);
        lp.iter(r,i) = iter;
        auc_stat(r, i) = auc;
    end
end

save('lp-binary-5.mat' ,'lp');



