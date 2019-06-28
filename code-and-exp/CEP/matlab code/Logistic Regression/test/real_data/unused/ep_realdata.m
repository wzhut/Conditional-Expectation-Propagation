addpath(genpath('../..'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% EP 
ep.ll = struct('mean', {}, 'std', {});
ep.kl = zeros(num_dataset,num_run);
ep.auc = zeros(num_dataset,num_run);

cfg.tol = 1e-5;
cfg.rho = 0.005;
cfg.max_iter = 5000;

for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        [logl, KL, auc, logls, KLs, aucs, time]= lrep(train,test, ts_mean, ts_var, cfg);
        ep.ll(i, nfold) = logl;
        ep.kl(i, nfold) = KL;
        ep.auc(i, nfold) = auc;
    end
end

save('ep-realdata.mat', 'ep');



