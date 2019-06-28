function [ auc, iter, aucs] = s_bcpep( train, test, rank, dim)





% train 
N = size(train,1);
threshold = 1e-3;

iter = 1;
max_iter = 500;
aucs = zeros([max_iter, 1]);
last_auc = 0;
last_diff = 0;

mean_list = zeros([rank, 1, 3, N]);
precision_list = reshape(repmat(reshape(eye(rank) * 1e-6, [rank*rank, 1]), [N * 3, 1]) , [rank, rank, 3, N]);

y = train(:,4);

joint = binitialize(train, dim, rank);

while true
    diff = 0;
    for i = 1 : N
        q_precision = zeros([rank, rank, 3]);
        q_mean = zeros([rank,1, 3]);
        new_precision = zeros([rank, rank, 3]);
        new_mean = zeros([rank, 1, 3]);
        for m = 1 : 3
            index = train(i,m);
            joint_precision = joint{m}.precision(:,:,index);
            joint_mean = joint{m}.mean(:,index);
            q_precision(:,:,m) = joint_precision - precision_list(:,:,m,i);
            q_mean(:,:,m) = q_precision(:,:,m) \ (joint_precision * joint_mean - precision_list(:,:,m,i) * mean_list(:,:,m,i));
        end
        
        exx = multinv(q_precision) + multiprod(q_mean, permute(q_mean, [2 1 3]));
        
        for m = 1 : 3
            index = setdiff(1:3, m);
            t = ones([rank,1]);
            tt = ones([rank, rank]);
            for k = 1 : length(index)
                t = t .* q_mean(:,:,index(k));
                tt = tt .* exx(:,:,index(k));
            end
            trtvt = trace(q_precision(:,:,m)\ tt);
            
            num = (2 * y(i) - 1) * t' * q_mean(:,:,m);
            den = (1 + trtvt) ^ 0.5;
            
            cdf = normcdf(num / den);
            pdf = normpdf(num / den);
            
            dmu = pdf / cdf / den * (2 * y(i) - 1) * t;
            dsig = pdf / cdf / den^3 * (0.5 - y(i)) * t' * q_mean(:,:,m) * tt;
            
            A = dmu * dmu' - 2 * dsig;
            var_m = inv(q_precision(:,:,m));
            
            tmp = inv(var_m - var_m * A * var_m);

            new_precision(:,:,m) = tmp - q_precision(:,:,m);
            [~, p] = chol(new_precision(:,:,m));
            if p > 0
                new_precision(:,:,m) = eye(rank) * 1e-6;
                tmp = new_precision(:,:,m) +  q_precision(:,:,m);
            end
            new_mean(:,:,m) = new_precision(:,:,m) \ (tmp * (q_mean(:,:,m) + var_m * dmu) - q_precision(:,:,m) * q_mean(:,:,m));
        end
        
        for m = 1 : 3
            index = train(i,m);
            joint_precision = q_precision(:,:,m) + new_precision(:,:,m);
            joint_mean = joint_precision \ (q_precision(:,:,m) * q_mean(:,:,m) + new_precision(:,:,m) * new_mean(:,:,m));
            
            diff = diff + mean(abs(joint{m}.mean(:,index) - joint_mean).^2);
            joint{m}.precision(:,:,index) = joint_precision;
            joint{m}.mean(:,index) = joint_mean;
        end
        
        precision_list(:,:,:,i) = new_precision;
        mean_list(:,:,:,i) = new_mean;
    end
    
    diff = diff / N / 3
    
    % train error
    
%     N1 = size(train_error,1);
    mu = ones([rank, N]);
    var = ones([rank, rank, N]);
    for m = 1: 3
        ids = train(:,m);
        mu = mu .* joint{m}.mean(:,ids);
        var = var .* multinv(joint{m}.precision(:,:,ids));
    end
    smu = sum(mu)';
    svar = reshape(sum(sum(var, 2), 1), [N, 1]);
    predicted_prob = normcdf(smu./sqrt(1 + svar));
%     predicted_prob = normcdf(smu);
    [~,~,~,auc] = perfcurve(train(:,4),predicted_prob,1);
%     auc = auc / N;
    aucs(iter) = auc;
    if abs(auc - last_auc) < threshold
%     if auc < threshold
        break;
    end
    if iter > max_iter
        break;
    end
    
    disp(sprintf('iter:%d  auc:%.8f',iter,auc)); 
    iter = iter + 1;
    last_auc = auc;
end
% test
N = size(test,1);

 mu = ones([rank, N]);
 var = ones([rank, rank, N]);
 for m = 1: 3
        ids = test(:,m);
        mu = mu .* joint{m}.mean(:,ids);
        var = var .* multinv(joint{m}.precision(:,:,ids));
end
smu = sum(mu)';
svar = reshape(sum(sum(var, 2), 1), [N, 1]);
predicted_prob = normcdf(smu./sqrt(1 + svar));
% predicted_prob = normcdf(smu);
[~,~,~,auc] = perfcurve(test(:,4),predicted_prob,1);


end
