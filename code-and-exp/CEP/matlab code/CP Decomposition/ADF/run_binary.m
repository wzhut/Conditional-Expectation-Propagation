rng(0);
rank_list = [3 5 8 10];
batch_size = [500 1000 500];
n = 5;
% continuous
bcpwopt_stats = zeros(4, 5, 3);
auc_stat = zeros(4, 5, 3);
time = zeros(4, 5, 3);

dim = [203 203 200]; 
cfg.tol = 1e-5;
cfg.rho = 0.5;
cfg.max_iter = 500;
cfg.verbose = 0;

for nfold = 2 : n
    train = importdata(sprintf('../complemented_data/train-fold-%d.txt',nfold),',');
    test = importdata(sprintf('../complemented_data/test-fold-%d.txt',nfold),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
        for k = 1 : length(batch_size)
            b = batch_size(k);
            disp(sprintf('nfold: %d rank: %d, batch_size: %d', nfold, rank, b));
            tic;
            [auc, iter, aucs] = b_cpadf(train, test, rank, dim, b, cfg);
            time(r, nfold, k) = toc;
            auc_stat(r, nfold, k) = auc;
        end
         
    end
end
save('binary.mat');
