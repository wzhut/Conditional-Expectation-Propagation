addpath(genpath('../..'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
rho = {0.1, 0.01, 0.1, 0.01, 0.01, 0.01};
cep_iter = {10, 50, 10, 50, 50, 50};
num_dataset = size(data_set,2);
num_run = 5;

% CEP2 
cep2.ll = zeros(num_dataset,num_run);
cep2.kl = zeros(num_dataset,num_run);
cep2.auc = zeros(num_dataset,num_run);
cep2.time = zeros(num_dataset, num_run);
cep2.lls = cell(num_data, num_run);

for i = 1:num_dataset    
    cfg.tol = 1e-5;
    cfg.max_iter = 5000;
    cfg.rho = rho{i};
    cfg.cep_iter = cep_iter{i};
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        tic;
        [logl, KL, auc, logls, KLs, aucs, time]= lrcep2_diag(train,test, ts_mean, ts_var, cfg);
        cep2.time(i, nfold) = toc;
        cep2.ll(i, nfold) = logl;
        cep2.kl(i, nfold) = KL;
        cep2.auc(i, nfold) = auc;
        cep2.lls{i, nfold} = logls;
    end
end

save('cep2-realdata.mat', 'cep2');



