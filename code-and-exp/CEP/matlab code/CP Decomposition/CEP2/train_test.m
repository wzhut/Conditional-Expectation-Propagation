k = 1;
rank = 3;
stats = cell([k, 1]);
rng(0);

cpwopt_stats=cell([k, 1]);

is_binary = true;

for i=1:k
        if is_binary
            train = importdata(sprintf('../complemented_data/train-fold-%d.txt',i),',');
            test = importdata(sprintf('../complemented_data/test-fold-%d.txt',i),',');
        else
            train = importdata(sprintf('../regression200x100x200/train-fold-%d.txt',i),',');
            test = importdata(sprintf('../regression200x100x200/test-fold-%d.txt',i),',');
        end
     
        
        if is_binary              
              [auc, iter, aucs] = bcpcep(train, test, rank, [203 203 200]);
              stats{i}.auc = auc;
              stats{i}.iter =iter;
              stats{i}.aucs = aucs;
        else            
            [mse, rmse, iter, diffs] = cpcep(train, test, rank, [200 100 200]);
            stats{i}.mse = mse;
            stats{i}.rmse = rmse;
            stats{i}.iter = iter;
            stats{i}.diffs = diffs;
       end


end


