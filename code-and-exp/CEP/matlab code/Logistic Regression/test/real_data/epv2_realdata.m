addpath(genpath('../..'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% EP v2
epv2.ll = zeros(num_dataset,num_run);
epv2.kl = zeros(num_dataset,num_run);
epv2.auc = zeros(num_dataset,num_run);
epv2.time = zeros(num_dataset,num_run);
epv2.lls = cell(num_dataset, num_run);

cfg.tol = 1e-5;
cfg.rho = 0.1;
cfg.max_iter = 5000;

for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        tic;
        [logl, KL, auc, logls, KLs, aucs, time]= lrepv2(train,test, ts_mean, ts_var, cfg);
        epv2.time(i, nfold) = toc;
        epv2.ll(i, nfold) = logl;
        epv2.kl(i, nfold) = KL;
        epv2.auc(i, nfold) = auc;
        epv2.lls{i, nfold} = logls;
    end
end

save('epv2-realdata.mat', 'epv2');



