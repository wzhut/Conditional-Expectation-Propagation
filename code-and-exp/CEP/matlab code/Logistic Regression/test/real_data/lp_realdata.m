addpath(genpath('../..'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% lp
lp.ll = zeros(num_dataset,num_run);
lp.kl = zeros(num_dataset,num_run);
lp.auc = zeros(num_dataset,num_run);
lp.time = zeros(num_dataset,num_run);
lp.lls = cell(num_dataset, num_run);

cfg.rho =0.1;
cfg.max_iter = 5000;
cfg.tol = 1e-5;

for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        tic;
        [logl, KL, auc, logls, KLs, aucs, time]= lrlp(train,test, ts_mean, ts_var, cfg);
        lp.time(i, nfold) = toc;
        lp.ll(i, nfold) = logl;
        lp.kl(i, nfold) = KL;
        lp.auc(i, nfold) = auc;
        lp.lls{i, nfold} = logls;
    end
end

save('lp-realdata.mat', 'lp');



