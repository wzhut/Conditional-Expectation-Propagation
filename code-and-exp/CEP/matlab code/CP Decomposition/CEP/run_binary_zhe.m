clear;

rng(0);
rank_list = [3 5 8 10];
n = 5;

auc = zeros(4, 5);
run_time = zeros(4, 5);
cep.time_stat = cell(4,5);
cep.test_auc_stat = cell(4,5);
cep.iter = zeros(4, 5);

nvec = [203 203 200];
cfg.tol = 1e-4;
cfg.rho = 0.05;
cfg.max_iter = 500;
cfg.verbose = 1;
for l = 2:2%length(rank_list)
    for nfold = 1 : n
        train = importdata(sprintf('../complemented_data/train-fold-%d.txt',nfold),',');
        test = importdata(sprintf('../complemented_data/test-fold-%d.txt',nfold),',');
        dim = rank_list(l);
        tic,
        %[U_invS_g, U_invSMean_g] = b_cpcep_zhev3(train, test, dim, nvec, cfg);
        [U_invS_g, U_invSMean_g] = b_cpcep_zhe_order2(train, test, dim, nvec, cfg);
        run_time(l,nfold) = toc;
        %test
        nmod = length(nvec);
        pred = ones(dim, 1, size(test,1));
        pred_var = ones(dim, dim, size(test,1));
        U_cov_g = cell(nmod,1);
        U_mean_g = cell(nmod,1);

        for k=1:nmod
            U_cov_g{k} = multinv(U_invS_g{k});
            U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
            pred = pred.*U_mean_g{k}(:,:,test(:,k));
            pred_var = pred_var.*U_cov_g{k}(:,:,test(:,k));
        end
        pred = sum(reshape(pred,[dim,size(test,1)]),1)';
        pred_var = sum(reshape(pred_var, [dim*dim, size(test,1)]),1)';
        prob = normcdf(pred./sqrt(1 + pred_var));
        %prob = pred;
        [~,~,~,auc(l, nfold)] = perfcurve(test(:,end),prob,1);
        
        fprintf('rank = %d, fold = %d, time = %g, auc = %g\n', dim, nfold, run_time(l, nfold), auc(l, nfold));
    end
end

res = [];
res.time = run_time;
res.auc = auc;
save('enron-cep-500.mat', 'res');


