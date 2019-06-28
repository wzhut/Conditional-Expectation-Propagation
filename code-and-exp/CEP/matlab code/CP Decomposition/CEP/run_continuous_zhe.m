clear all;
close all;
addpath_recurse('../../../../lightspeed');
addpath_recurse('./mtimesx_20110223');
addpath_recurse('./multinv');
rng(0);
rank_list = [3 5 8 10];
% continuous

mse_stat = zeros(4, 5);
rmse_stat = zeros(4 ,5);
time = zeros(4, 5);
dim = [200 100 200];
cfg.tol = 0;
cfg.rho = 0.1;
cfg.max_iter = 500;
cfg.verbose = 1;
rmse = zeros(length(rank_list), 5);
run_time = zeros(length(rank_list), 5);
for l =2:2%length(rank_list)
    R = rank_list(l);
    for nfold = 1 : 5
        train = importdata(sprintf('../regression200x100x200/train-fold-%d.txt',nfold),',');
        test = importdata(sprintf('../regression200x100x200/test-fold-%d.txt',nfold),',');
        tic,
        [U_invS_g, U_invSMean_g, tau] = c_cpcep_zhev2(train, test, R, cfg);
        run_time(l, nfold) = toc;

        %test
        nmod = size(train,2)-1;
        pred = ones(R, 1, size(test,1));
        U_cov_g = cell(nmod,1);
        U_mean_g = cell(nmod,1);
        for k=1:nmod
            U_cov_g{k} = multinv(U_invS_g{k});
            U_mean_g{k} = mtimesx(U_cov_g{k}, U_invSMean_g{k});                
            pred = pred.*U_mean_g{k}(:,:,test(:,k));
        end
        pred = sum(reshape(pred,[R,size(test,1)]),1)';
        rmse(l,nfold) = sqrt(mean((pred - test(:,end)).^2));
        fprintf('rank =  %d, fold %d, time =%g, rmse = %g\n', R, nfold, run_time(l, nfold), rmse(l, nfold));
    end

end
res = [];
res.time = run_time;
res.rmse = rmse;
save('alog-cep-500.mat', 'res');