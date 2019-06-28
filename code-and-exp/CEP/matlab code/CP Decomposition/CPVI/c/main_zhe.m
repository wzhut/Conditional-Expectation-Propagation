% % % ACC batch
clear all;
close all;

% addpath('tensor_toolbox_2.5');
rng('default');

rank_list = [3 5 8 10];
mse = zeros(4,5);
rmse = zeros(4,5);
run_time = zeros(4,5);

vi.time_stat = cell(4, 5);
vi.rmse_stat = cell(4, 5);
vi.iter = zeros(4,5);


for l = 1 : 1%length(rank_list)    
    for nfold = 1 : 5
        train = importdata(sprintf('../../regression200x100x200/train-fold-%d.txt',nfold),',');
        test = importdata(sprintf('../../regression200x100x200/test-fold-%d.txt',nfold),',');
        batch_size = size(train, 1);
        nvec = [200 100 200];
        nmod = length(nvec);
        post_mean = cell(nmod,1);
        post_cov = cell(nmod, 1);
        %settings
        opt = [];
        opt.max_iter = 500;
        opt.tol = 0;
        %3,5,8,10
        R = rank_list(l);
        v = 1;
        tic,
        for k=1:nmod
            post_mean{k} = rand(nvec(k),R);
            post_cov{k} = reshape(repmat(v*eye(R), [1, nvec(k)]), [R,R,nvec(k)]);
        end
        post = {post_mean, post_cov, [0.001, 0.001]};

        for i=1:batch_size:size(train,1)
            if i+batch_size-1<=size(train,1)
                batch_data = train(i:i+batch_size-1, :);
            else
                batch_data = train(i:end,:);
            end
            batch_data(:,1:nmod) = batch_data(:,1:nmod);
            post = POST_cont_zhe(post, batch_data, opt, test);
        end
        run_time(l, nfold) = toc;

        post_mean = post{1};
        post_cov = post{2};
        true_val = test(:,4);
        test_ind = test(:,1:3);
        n_test = size(test_ind,1);
        mu = ones(n_test,R);
        for k=1:nmod
            mu = mu.*post_mean{k}(test_ind(:,k),:);
        end
        pred = sum(mu,2);
        rmse(l, nfold) = sqrt(mean((true_val - pred).^2));    
        fprintf('rank =  %d, fold %d, time =%g, rmse = %g\n',R, nfold, run_time(l, nfold), rmse(l, nfold));      
    end
end
res = [];
res.time = run_time;
res.rmse = rmse;
save('alog.mat', 'res');






