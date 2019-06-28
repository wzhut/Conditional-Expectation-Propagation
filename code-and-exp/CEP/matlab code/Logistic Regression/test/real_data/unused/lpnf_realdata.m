addpath(genpath('../..'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% lp nf
lpnf.ll = struct('mean', {}, 'std', {});
lpnf.kl = zeros(num_dataset,num_run);
lpnf.auc = zeros(num_dataset,num_run);

cfg.rho =0.005;
cfg.max_iter = 5000;
cfg.tol = 1e-5;

for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        [logl, KL, auc, logls, KLs, aucs, time]= lrlp_nf(train,test, ts_mean, ts_var, cfg);
        lpnf.ll(i, nfold) = logl;
        lpnf.kl(i, nfold) = KL;
        lpnf.auc(i, nfold) = auc;
    end
end

save('lpnf-realdata.mat', 'lpnf');



