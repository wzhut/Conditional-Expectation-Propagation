clear;
addpath(genpath('./multinv'));
addpath(genpath('./mtimesx_20110223'));
addpath_recurse('../../../../lightspeed');
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
        [~, ~, ~, iter, test_rmses, time] = c_cpcep_v2(train, test, rank, cfg);
%         time(r, nfold) = toc;
        cep.rmse_stat{r, nfold} = test_rmses;
        cep.time_stat{r, nfold} = cumsum(time);
        cep.iter(r, nfold) = iter;
    end
end

save('cep-continuous-performance.mat', 'cep');

