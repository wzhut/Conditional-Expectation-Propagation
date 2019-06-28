clear;
addpath(genpath('./multinv'));
addpath(genpath('./mtimesx_20110223'));
addpath(genpath('./lightspeed'));
rng(0);
rank_list = [8];
batch_size = 5000;
n = 3;
% continuous
nmod = 3;
% 0 indexed
train = load('../tensor-data-large/dblp/dblp-large-tensor.txt');
test = load('../tensor-data-large/dblp/dblp.mat');
test = test.data.test;

train(:,1:3) = train(:,1:3) + 1;
% test(:,1:3) = test(:,1:3) + 1;

dim = [10000,200,10000]; 

cfg.tol = 1e-3;
cfg.rho = 0.1;
cfg.max_iter = 500;
cfg.verbose = 0;
rank = 8;

dummy_test = [test{1}.subs test{1}.Ymiss];
for nfold = 1 : n
    train = train(randperm(size(train,1)),:);
end
    disp(sprintf('nfold: %d rank: %d', nfold, rank));
    tic;
    [U_invS_g, U_invSMean_g] = b_cpadf_v2(train, dummy_test, rank, dim, batch_size, cfg);
    time = toc;
    avg_auc = 0;
    for i = 1 : 50
        cur_batch = [test{i}.subs test{i}.Ymiss];
        pred = ones(rank, 1, size(cur_batch,1));
        pred_var = ones(rank, rank, size(cur_batch,1));
        U_cov_g = cell(nmod,1);
        U_mean_g = cell(nmod,1);

        for k=1:nmod
            U_cov_g{k} = multinv(U_invS_g{k});
            U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
            pred = pred.*U_mean_g{k}(:,:,cur_batch(:,k));
            pred_var = pred_var.*U_cov_g{k}(:,:,cur_batch(:,k));
        end
        pred = sum(reshape(pred,[rank,size(cur_batch,1)]),1)';
        pred_var = sum(reshape(pred_var, [rank*rank, size(cur_batch,1)]),1)';
        prob = normcdf(pred./sqrt(1 + pred_var));
        %prob = pred;
        [~,~,~,auc] = perfcurve(cur_batch(:,end),prob,1);
        avg_auc = avg_auc + auc;
    end
    avg_auc = avg_auc / 50;
    auc = avg_auc;
save('dblp_r8_b5k-3.mat');
