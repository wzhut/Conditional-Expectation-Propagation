rng(0);
rank_list = [1];%[3 5 8 10];
n = 5;
% continuous
bcpwopt_stats = zeros(5, 5);
bstats = zeros(5, 5);

for nfold = 1 : n
    train = importdata(sprintf('../complemented_data/train-fold-%d.txt',nfold),',');
    test = importdata(sprintf('../complemented_data/test-fold-%d.txt',nfold),',');
    for r = 1: length(rank_list)
        rank = rank_list(r);
%         trind = train(:,1:3);
%         trval = train(:,4);
%         dim = [203 203 200]; 
%         Y = sptensor(trind, trval, dim);
%         W = sptensor(trind, ones(size(trind,1), 1), dim);
%         K = cp_wopt(Y, W,rank);
%         T = sptensor(tensor(K));
%         predicted_prob = T(test(:,1:3));
%         [~,~,~,auc] = perfcurve(test(:,4), predicted_prob, 1);
%         bcpwopt_stats(r, nfold, 1) = auc;
        disp(sprintf('nfold: %d rank: %d', nfold, rank));
        [auc, iter, aucs] = bcpcep(train, test, rank, [203 203 200]);
        bstats(r, nfold) = auc;
    end
end

