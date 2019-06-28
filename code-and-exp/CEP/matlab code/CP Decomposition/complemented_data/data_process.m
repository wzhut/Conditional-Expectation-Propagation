rng(0);
for i = 1 : 5
 train = importdata(sprintf('./enron_fold - %d_train.csv',i),',');
 test = importdata(sprintf('./enron_fold - %d_test.csv',i),',');
 
    trind = train(:,1:3);
    subs = test(:,1:3);
    nevc = [203 203 200];
    Y = sptensor(trind, ones(length(trind),1), nvec);
    zind = tt_sub2ind(size(Y), find(Y==0));
    zind = setdiff(zind, tt_sub2ind(size(Y), subs));%remove test 0s
    zind = tt_ind2sub(size(Y), zind);
    zind_sample = datasample(zind, 1*length(trind),'Replace', false);
    zind_sample(:,4) = 0;
    train = [train; zind_sample];
    
 csvwrite(sprintf('./train-fold-%d.txt',i), train);
 csvwrite(sprintf('./test-fold-%d.txt',i), test);
end