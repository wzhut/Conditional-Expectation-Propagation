k = 1;
rank = 3;
stats = cell([k, 1]);
rng(0);

cpwopt_stats=cell([k, 1]);

is_binary = false;

for i=1:k
%     for rank = 10:20
        if is_binary
            train = importdata(sprintf('./complemented_data/train-fold-%d.txt',i),',');
            test = importdata(sprintf('./complemented_data/test-fold-%d.txt',i),',');
        else
            train = importdata(sprintf('./regression200x100x200/train-fold-%d.txt',i),',');
            test = importdata(sprintf('./regression200x100x200/test-fold-%d.txt',i),',');
        end
       
        trind = train(:,1:3);
        trval = train(:,4);
        dim = [210 210 210];
        Y = sptensor(trind, trval, dim);
        W = sptensor(trind, ones(size(trind,1), 1), dim);
        K = cp_wopt(Y, W,rank);
        
        if is_binary
              T = sptensor(tensor(K));
              predicted_prob = T(test(:,1:3));
              [~,~,~,auc] = perfcurve(test(:,4), predicted_prob, 1);
              cpwopt_stats{i}.auc = auc;
              
              [auc, iter, aucs] = b_bcpep(train, test, rank, [203 203 200]);
%               [auc, iter, aucs] = s_bcpep(train, test, rank, [203 203 200]);
              stats{i}.auc = auc;
              stats{i}.iter =iter;
              stats{i}.aucs = aucs;
          else
            predicted = ones(size(test,1),rank);
            for m = 1: 3
                idx = test(:,m);
                predicted = predicted .* K.U{m}(idx,:);
            end
            mse = mean((sum(predicted,2) - test(:,4)).^2);
            rmse = sqrt(mse);
            cpwopt_stats{i}.mse = mse;
            cpwopt_stats{i}.rmse = rmse;
            
            [mse, rmse, iter, diffs] = cpep(train, test, rank, [200 100 200]);
            stats{i}.mse = mse;
            stats{i}.rmse = rmse;
            stats{i}.iter = iter;
            stats{i}.diffs = diffs;
       end


end

if is_binary
    for i=1:5
        subplot(k,1,i);
        plot(1:stats{i}.iter, stats{i}.aucs(1:stats{i}.iter));
    end
else
    for i=1:5
        subplot(k,1,i);
        plot(1:stats{i}.iter, stats{i}.diffs(1:stats{i}.iter));
    end
end


