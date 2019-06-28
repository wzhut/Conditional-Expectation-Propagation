k = 5;
rank_list = [3 5 8 10];
stats = cell([k, 1]);
rng(0);

cpwopt_stats=cell([k, 4]);

is_binary = true;

for i=1:k
    for r = 1: length(rank_list)
        if is_binary
            train = importdata(sprintf('../complemented_data/train-fold-%d.txt',i),',');
            test = importdata(sprintf('../complemented_data/test-fold-%d.txt',i),',');
        else
            train = importdata(sprintf('../regression200x100x200/train-fold-%d.txt',i),',');
            test = importdata(sprintf('../regression200x100x200/test-fold-%d.txt',i),',');
        end
       rank = rank_list(r);

        
        if is_binary   
              [auc, iter, aucs] = b_cplp(train, test, rank, [203 203 200]);
%               [auc, iter, aucs] = s_bcpep(train, test, rank, [203 203 200]);
              stats{i,r}.auc = auc;
              stats{i,r}.iter =iter;
              stats{i,r}.aucs = aucs;
        else
            [mse, rmse, iter, diffs] = c_cplp(train, test, rank, [200 100 200]);
            stats{i,r}.mse = mse;
            stats{i,r}.rmse = rmse;
            stats{i,r}.iter = iter;
            stats{i,r}.diffs = diffs;
        end
    end


end



