% % % ACC batch
clear all;
close all;

% addpath('tensor_toolbox_2.5');
rng('default');

rank_list = [3 5 8 10];
mse = zeros(4,5);
rmse = zeros(4,5);
time = zeros(4,5);

vi.time_stat = cell(4, 5);
vi.rmse_stat = cell(4, 5);
vi.iter = zeros(4,5);

diary output/diary

for nfold = 1 : 5
    for l = 1 : length(rank_list)
        train = importdata(sprintf('../../regression200x100x200/train-fold-%d.txt',nfold),',');
        test = importdata(sprintf('../../regression200x100x200/test-fold-%d.txt',nfold),',');
        %1,5,10,50,100,1K,5K,10K
        batch_size = size(train, 1);
        nvec = [200 100 200];
        nmod = length(nvec);
        post_mean = cell(nmod,1);
        post_cov = cell(nmod, 1);
        %settings
        opt = [];
        opt.max_iter = 500;
        opt.tol = 1e-4;
        %3,5,8
        R = rank_list(l);
        v = 1;
        
        nruns = 1;
        mse_folds = zeros(nruns,1);
        for fold=1:nruns
            tic;
            fprintf('run %d, batch size %d , R = %d \n',fold, batch_size,R);
            train = train(randperm(size(train,1)),:);
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
                
                [post_u, iter, test_rmses, time_stat] = POST_cont(post, batch_data, opt, test);
                
                vi.time_stat{l, nfold} = cumsum(time_stat);
                vi.rmse_stat{l,nfold} = test_rmses;
                vi.iter(l,nfold) = iter;
                
                if mod(i-1, 1000*batch_size) == 0
                %if mod(i-1, 1*batch_size) == 0
                    fprintf('%d batches processed!\n', i);
                end
            end
            toc
            time(l, nfold) = time(l, nfold) + toc;

%             test = load ('./tensor-data-large/acc.mat');
%             test = test.data.test;
            post_mean = post{1};
            post_cov = post{2};
%             for i=1:50
                true_val = test(:,4);
                test_ind = test(:,1:3);
                n_test = size(test_ind,1);
                mu = ones(n_test,R);
                for k=1:nmod
                    mu = mu.*post_mean{k}(test_ind(:,k),:);
                end
                pred = sum(mu,2);
                err = mean((true_val - pred).^2);    
                avg_err = err;
                mse(l,nfold) = mse(l,nfold) + err;
                rmse(l, nfold) = rmse(l, nfold) + sqrt(err);
%             end
            fprintf('run %d, mse = %g\n',fold, avg_err);
            mse_folds(fold) = avg_err;

            fn_U = strcat('./output/zsd-time-',int2str(fold),'-R-',int2str(R),'-V-',num2str(v),'-U','-batch-',int2str(batch_size),'.mat'); 
            save(fn_U, 'post'); %%%%%%%%
            toc;

        end
        fprintf('5 Runs: average mse = %g, std = %g\n', mean(mse_folds), std(mse_folds)/sqrt(nruns)); 
    end
end

mse = mse / 5;
rmse = rmse / 5;

save('vi-continuous-performance.mat');

% train = load ('./tensor-data-large/acc-large-tensor.txt');


diary off



