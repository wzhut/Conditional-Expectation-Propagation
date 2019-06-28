clear all;
close all;
% addpath_recurse('../../Research/LargeTensor/InfTucker-DP-Large2/tensor_toolbox_2.5');

rng('default');
auc = zeros(5,1);
dim = 3;
stats = cell(5,1);
for nfold = 1:5
    filename =['./small-data/enron_fold_', num2str(nfold), '.mat']; 
    dt = load(filename);
    dt = dt.dt;
    trind = dt.train(:,1:3);
    trvals = dt.train(:,end);
    nvec = [203 203 200];
    W = sptensor(trind, ones(length(trind),1), nvec);

    subs = dt.test(:,1:3);
    ymiss = dt.test(:,end);
    Y = sptensor(trind, ones(length(trind),1), nvec);
    zind = tt_sub2ind(size(Y), find(Y==0));
    zind = setdiff(zind, tt_sub2ind(size(Y), subs));%remove test 0s
    zind = tt_ind2sub(size(Y), zind);
    zind_sample = datasample(zind, 1*length(trind),'Replace', false);
%     subs = setdiff(subs, zind_sample, 'rows');
    W(zind_sample) = ones(length(zind_sample),1);
    P = cp_wopt(Y,W,dim);
    M = sptensor(tensor(P));
    est=M(subs);
    [~,~,~,val] = perfcurve(ymiss, est, 1);
    fprintf('fold %d, auc = %g\n', nfold,  val);
    auc(nfold) = val;
    zind_sample(:,4) = 0;
    train = [dt.train; zind_sample];
    test = dt.test;
    csvwrite(sprintf('./train-fold-%d.txt',nfold), train);
    csvwrite(sprintf('./test-fold-%d.txt',nfold), test);
%     rank = 3;
%     [auc, iter, aucs] = s_bcpep(train, test, rank, [203 203 200]);
%     stats{nfold}.auc = auc;
%     stats{nfold}.iter =iter;
%     stats{nfold}.aucs = aucs;
end
auc'