clear;

rng(0);
rank_list = [3 5 8 10];
n = 5;

auc_stat = zeros(4, 5);
% time = zeros(4, 5);
cep.time_stat = cell(4,5);
cep.test_auc_stat = cell(4,5);
cep.iter = zeros(4, 5);

dim = [203 203 200];
cfg.tol = 1e-4;
cfg.rho = 0.5;
cfg.max_iter = 500;
cfg.verbose = 1;

for nfold = 1 : n
    train = importdata(sprintf('../complemented_data/train-fold-%d.txt',nfold),',');
    test = importdata(sprintf('../complemented_data/test-fold-%d.txt',nfold),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
        disp(sprintf('nfold: %d rank: %d', nfold, rank));
%         tic;
        [auc, iter,test_aucs, time] = b_cpcep(train, test, rank, dim, cfg);
%         time(r, nfold) = toc;
        cep.time_stat{r, nfold} = cumsum(time);
        cep.test_auc_stat{r, nfold} = test_aucs;
        cep.iter(r, nfold) = iter;
        auc_stat(r, nfold) = auc;
    end
end
save('cep-binary.mat', 'cep');
