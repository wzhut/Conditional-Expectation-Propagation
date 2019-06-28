addpath(genpath('../..'));
rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% VB nf
vbnf.ll = struct('mean', {}, 'std', {});
vbnf.kl = zeros(num_dataset,num_run);
vbnf.auc = zeros(num_dataset,num_run);


for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        [logl, KL, auc, logls, KLs, aucs, time]= prvb_nf(train,test, ts_mean, ts_var);
        vbnf.ll(i, nfold) = logl;
        vbnf.kl(i, nfold) = KL;
        vbnf.auc(i, nfold) = auc;
    end
end

save('vbnf-realdata.mat', 'vbnf');



