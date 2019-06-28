addpath(genpath('../..'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
rho = {0.1, 0.01, 0.1, 0.01, 0.01, 0.01};
num_dataset = size(data_set,2);
num_run = 5;

% CEP 
cep.ll = zeros(num_dataset,num_run);
cep.kl = zeros(num_dataset,num_run);
cep.auc = zeros(num_dataset,num_run);
cep.time = zeros(num_dataset, num_run);
cep.lls = cell(num_dataset, num_run);



for i = 1:num_dataset    
    cfg.tol = 1e-5;
    cfg.rho = rho{i};
    cfg.max_iter = 5000;
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        tic;
        [logl, KL, auc, logls, KLs, aucs, time]= prcep(train,test, ts_mean, ts_var, cfg);
        cep.time(i, nfold) = toc;
        cep.ll(i, nfold) = logl;
        cep.kl(i, nfold) = KL;
        cep.auc(i, nfold) = auc;
        cep.lls{i, nfold} = logls;
    end
end

save('cep-realdata.mat', 'cep');



