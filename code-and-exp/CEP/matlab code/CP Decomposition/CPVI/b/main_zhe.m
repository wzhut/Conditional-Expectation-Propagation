% % DBLP batch
clear;
clc;

% addpath('tensor_toolbox_2.5');

rank_list = [3 5 8 10];
auc_stat = zeros(4,5);
time = zeros(4, 5);
rng('default');

for nfold = 2 : 5
    for l = 1 : length(rank_list)
        
    train = importdata(sprintf('../../complemented_data/train-fold-%d.txt',nfold),',');
    test = importdata(sprintf('../../complemented_data/test-fold-%d.txt',nfold),',');
    %1,5,10,50,100,1K,5K,10K
    batch_size = size(train,1);

    nvec = [203,203,200];
    nmod = length(nvec);
    post_mean = cell(nmod,1);
    post_cov = cell(nmod, 1);
    %settings
    opt = [];
    opt.max_iter = 500;
    opt.tol = 0;
    %rank 3,5,8
    R = rank_list(l);
    v = 1;
    
    % train = load ('./tensor-data-large/dblp-large-tensor.txt');
    nruns = 1;
    auc_folds = zeros(nruns,1);
    for fold=1:nruns
        
         fprintf('run %d, batch size %d ,R = %d \n',fold, batch_size,R);
        train = train(randperm(size(train,1)),:);
        tic,
        for k=1:nmod
            post_mean{k} = rand(nvec(k),R);
            post_cov{k} = reshape(repmat(v*eye(R), [1, nvec(k)]), [R,R,nvec(k)]);
        end
        post_u = {post_mean, post_cov};
        
        for i=1:batch_size:size(train,1)
            if i+batch_size-1<=size(train,1)
                batch_data = train(i:i+batch_size-1, :);
            else
                batch_data = train(i:end,:);
            end
            batch_data(:,1:nmod) = batch_data(:,1:nmod);
            [post_u] = POST_zhe(post_u,batch_data,opt, test);
            
        end

        time(l, nfold) = time(l, nfold) + toc;


        avg_auc = 0;
        post_mean = post_u{1};
        post_cov = post_u{2};
%         for i=1:50
            true_val = test(:,4);
            test_ind = test(:,1:3);
            n_test = size(test_ind,1);
            mu = ones(n_test,R);
            S = ones(R,R,n_test);
            for k=1:nmod
                mu = mu.*post_mean{k}(test_ind(:,k),:);
                S = S.*post_cov{k}(:, :, test_ind(:,k));
            end
            m = sum(mu,2);
            sq = squeeze(sum(sum(S,2),1));
            prob = normcdf(m./sqrt(1+sq)); % prob = normcdf(u/sqrt(1+sigma^2)) here, u := m
            [~,~,~,auc] = perfcurve(true_val,prob,1);
            auc_stat(l,nfold) = auc_stat(l,nfold) + auc;
            avg_auc = auc;
%         end
%         avg_auc = avg_auc/50;
        fprintf('run %d, auc = %g\n',fold, avg_auc);
        auc_folds(fold) = avg_auc;
    end
    fprintf('5 Runs: average auc = %g, std = %g\n', mean(auc_folds), std(auc_folds)/sqrt(nruns)); 
    end
end

auc_stat = auc_stat / nruns;
time = time / nruns;

save('b.mat', 'vi');
diary off



