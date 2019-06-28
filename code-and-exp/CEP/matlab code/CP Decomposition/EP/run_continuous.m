rng(0);
rank_list = [1 3 5 8 10];
n = 5;
% continuous
cpwopt_stats = zeros(5, 5, 2);
stats = zeros(5, 5, 2);

for nfold = 1 : n
    train = importdata(sprintf('./regression200x100x200/train-fold-%d.txt',nfold),',');
    test = importdata(sprintf('./regression200x100x200/test-fold-%d.txt',nfold),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
        trind = train(:,1:3);
        trval = train(:,4);
        dim = [200 100 200]; % [210 210 210];
        Y = sptensor(trind, trval, dim);
        W = sptensor(trind, ones(size(trind,1), 1), dim);
        K = cp_wopt(Y, W,rank);
        predicted = ones(size(test,1),rank);
        for m = 1: 3
             idx = test(:,m);
             predicted = predicted .* K.U{m}(idx,:);
        end
        mse = mean((sum(predicted,2) - test(:,4)).^2);
        rmse = sqrt(mse);
        cpwopt_stats(r, nfold, 1) = mse;
        cpwopt_stats(r, nfold, 2) = rmse;
        disp(sprintf('nfold: %d rank: %d', nfold, rank));    
        [mse, rmse, iter, diffs] = cpep(train, test, rank, dim);
        stats(r, nfold, 1) = mse;
        stats(r, nfold, 2) = rmse;
    end
end

