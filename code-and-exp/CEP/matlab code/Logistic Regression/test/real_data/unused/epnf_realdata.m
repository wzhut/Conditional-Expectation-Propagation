addpath(genpath('../..'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% EP nf
epnf.ll = struct('mean', {}, 'std', {});
epnf.kl = zeros(num_dataset,num_run);
epnf.auc = zeros(num_dataset,num_run);

cfg.tol = 1e-5;
cfg.rho = 0.005;
cfg.max_iter = 5000;

for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        [logl, KL, auc, logls, KLs, aucs, time]= lrep_nf(train,test, ts_mean, ts_var, cfg);
        epnf.ll(i, nfold) = logl;
        epnf.kl(i, nfold) = KL;
        epnf.auc(i, nfold) = auc;
    end
end

save('epnf-realdata.mat', 'epnf');



