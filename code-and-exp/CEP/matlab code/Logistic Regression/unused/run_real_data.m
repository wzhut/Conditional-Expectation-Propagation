addpath(genpath('./lightspeed'));
addpath(genpath('./minFunc_2012'));
addpath(genpath('./ghq'));

rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% CEP 
cepll = struct('mean', {}, 'std', {});
cepkl = zeros(num_dataset,num_run);
cepauc = zeros(num_dataset,num_run);
% CEP 2
cep2ll = struct('mean', {}, 'std', {});
cep2kl = zeros(num_dataset,num_run);
cep2auc = zeros(num_dataset,num_run);
% EP 
epll = struct('mean', {}, 'std', {});
epkl = zeros(num_dataset,num_run);
epauc = zeros(num_dataset,num_run);
% EPnf 
epnfll = struct('mean', {}, 'std', {});
epnfkl = zeros(num_dataset,num_run);
epnfauc = zeros(num_dataset,num_run);
% VB
vbll = struct('mean', {}, 'std', {});
vbkl = zeros(num_dataset,num_run);
vbauc = zeros(num_dataset,num_run);
% LP
lpll = struct('mean', {}, 'std', {});
lpkl = zeros(num_dataset,num_run);
lpauc = zeros(num_dataset,num_run);
% LPnf 
lpnfll = struct('mean', {}, 'std', {});
lpnfkl = zeros(num_dataset,num_run);
lpnfauc = zeros(num_dataset,num_run);

cfg.tol = 1e-5;
cfg.rho = 0.005;
cfg.max_iter = 5000;

for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('./data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');
        
        % CEP
        [logl, KL, auc, logls, KLs, aucs, time]= lrcep(train,test, ts_mean, ts_var, cfg);
        cepll(i, nfold) = logl;
        cepkl(i, nfold) = KL;
        cepauc(i, nfold) = auc;

        % CEP2 diagonal
        [logl, KL, auc, logls, KLs, aucs, time]= lrcep2_diag(train,test, ts_mean, ts_var, cfg);
        cep2ll(i, nfold) = logl;
        cep2kl(i, nfold) = KL;
        cep2auc(i, nfold) = auc;
        
        % EP
        [logl, KL, auc, logls, KLs, aucs, time]= lrep(train,test, ts_mean, ts_var, cfg);
        epll(i, nfold) = logl;
        epkl(i, nfold) = KL;
        epauc(i, nfold) = auc;
        
        % EP non-factorized
        [logl, KL, auc, logls, KLs, aucs, time]= lrep_nf(train,test, ts_mean, ts_var, cfg);
        epnfll(i, nfold) = logl;
        epnfkl(i, nfold) = KL;
        epnfauc(i, nfold) = auc;
        
        % VB
        [logl, KL, auc, logls, KLs, aucs, time]= lrvb(train,test, ts_mean, ts_var);
        vbll(i, nfold) = logl;
        vbkl(i, nfold) = KL;
        vbauc(i, nfold) = auc;
        
        % LP
        [logl, KL, auc, logls, KLs, aucs, time]= lrlp_nf(train,test, ts_mean, ts_var, cfg);
        lpnfll(i, nfold) = logl;
        lpnfkl(i, nfold) = KL;
        lpnfauc(i, nfold) = auc;
        
        % LP non-factorized
        [logl, KL, auc, logls, KLs, aucs, time]= lrlp(train,test, ts_mean, ts_var, cfg);
        lpll(i, nfold) = logl;
        lpkl(i, nfold) = KL;
        lpauc(i, nfold) = auc;

    end
end

save('real_data.mat');



