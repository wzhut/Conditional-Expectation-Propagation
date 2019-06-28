addpath(genpath('../..'));
rng(0);
data_set = {'australian', 'breast', 'crabs', 'ionos', 'pima', 'sonar'};
num_dataset = size(data_set,2);
num_run = 5;

% VB 
vb.ll = zeros(num_dataset,num_run);
vb.kl = zeros(num_dataset,num_run);
vb.auc = zeros(num_dataset,num_run);
vb.time = zeros(num_dataset, num_run);
vb.lls = cell(num_dataset, num_run);


for i = 1:num_dataset    
    for nfold = 1 : num_run
        load(sprintf('../data/%s/nfold-%d.mat', data_set{i}, nfold), 'train', 'test', 'ts_mean', 'ts_var');     
        % CEP
        tic;
        [logl, KL, auc, logls, KLs, aucs, time]= prvb(train,test, ts_mean, ts_var);
        vb.time(i, nfold) = toc;
        vb.ll(i, nfold) = logl;
        vb.kl(i, nfold) = KL;
        vb.auc(i, nfold) = auc;
        vb.lls{i, nfold} = logls;
    end
end

save('vb-realdata.mat', 'vb');



