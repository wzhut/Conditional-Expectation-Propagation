function [ auc, iter, aucs] = bcpep( train, test, rank, dim)

% train 
N = size(train, 1);
threshold = 0;
rho = 0.7;
iter = 1;
max_iter = 500;
aucs = zeros([max_iter, 1]);
last_auc = 0;
last_diff = 0;

mean_list = zeros([rank, 1, 3, N]);
precision_list = reshape(repmat(reshape(eye(rank) * 1e-6, [rank*rank, 1]), [N * 3, 1]) , [rank, rank, 3, N]);

y = train(:,4);


% q\
q_mean = zeros([rank, 1, 3, N]);
q_precision = zeros([rank, rank, 3, N]);

% initialize
joint = binitialize(train, dim, rank);

while true
    % get expectations from q\
    for i = 1 : N
        for m = 1 : 3
            id = train(i,m);
            joint_precision = joint{m}.precision(:,:,id);
            joint_mean = joint{m}.mean(:,id);
            q_precision(:,:,m,i) = joint_precision - precision_list(:,:,m,i);
            q_mean(:,:,m,i) = q_precision(:,:,m,i) \ (joint_precision * joint_mean - precision_list(:,:,m,i) * mean_list(:,:,m,i));
        end
    end
    
    exx = multinv(q_precision) + multiprod(q_mean, permute(q_mean, [2 1 3 4]));
    new_mean_list = zeros([rank, 1, 3, N]); % r 1 3 N
    new_precision_list = zeros([rank, rank, 3, N]); % r r 3 N
    for m = 1:3
        t = ones([rank, 1, 1, N]);
        tt = ones([rank, rank, 1, N]);
        var_m = multinv(q_precision(:,:,m,:));
        % trace + t var_ a t
        trtvt = zeros([N, 1]);
        index = setdiff(1:3, m);
        % t var_t
        for k = 1:length(index)
            t = t .* q_mean(:,:,index(k),:);
            tt = tt .* exx(:,:,index(k),:);
        end
        t_t = permute(t, [2 1 3 4]);
        tmp = multiprod(var_m, tt);
        % trace
        for k = 1 : N
            trtvt(k) = trace(tmp(:,:,1,k));
        end

        % numerator and denominator
        num = (2 * y - 1) .* reshape(multiprod(t_t, q_mean(:,:,m,:)), [N,1]);
        den = (1 + trtvt).^(0.5);
        
        cdf = arrayfun(@(a, b) normcdf(a/b) + realmin, num, den, 'UniformOutput', true);
        pdf = arrayfun(@(a, b) normpdf(a/b) + realmin, num, den, 'UniformOutput', true);
        
        % partial derivatives
        dmu = pdf ./ cdf ./ den .* (2 * y - 1);
%         arrayfun(@(a, b, c, d) (2*b - 1) * d / (a * c), cdf, y, den, pdf, 'UniformOutput', true);
        dmu = bsxfun(@times, reshape(dmu, [1,1,1 N]), t);
        
        dsig = pdf ./ cdf ./ den.^3 .* (0.5 - y); 
%         arrayfun(@(a, b, c, d) -0.5 * (2*b - 1) * d / (a * c^3), cdf, y, den, pdf, 'UniformOutput', true);
        tmp = multiprod(multiprod(t_t, q_mean(:,:,m,:)), tt);        
        dsig = bsxfun(@times, reshape(dsig, [1,1,1,N]), tmp);
        
        % calculate new values
        A = multiprod(dmu, permute(dmu, [2 1 3 4])) - 2 * dsig; % -2
        tmp = var_m - multiprod(var_m, multiprod(A, var_m));
        tmp = multinv(tmp);
        new_precision = tmp - q_precision(:,:,m,:);
        
        for k= 1 : N
           [~, p] = chol(new_precision(:,:,1,k));
           if p > 0
              new_precision(:,:,1,k) = eye(rank) * 1e-6;
              tmp(:,:,1,k) = q_precision(:,:,m,k) + new_precision(:,:,1,k);
           end
        end

        new_mean = multiprod(tmp, q_mean(:,:,m,:) + multiprod(var_m, dmu)) - multiprod(q_precision(:,:,m,:), q_mean(:,:,m,:));
        new_mean = multiprod(multinv(new_precision), new_mean);
        
        new_precision_list(:,:,m,:) = new_precision;
        new_mean_list(:,:,m,:) = new_mean;

    end
    
    % update
    update_mean_list = mean_list * (1 - rho) + new_mean_list * rho;
    update_precision_list = precision_list * (1 - rho) + new_precision_list * rho;
    

    diff = 0;
    for i = 1 : N
        for m = 1 : 3
            id = train(i,m);
            joint_precision = joint{m}.precision(:,:,id);
            joint_mean = joint{m}.mean(:,id);
            
            joint_mean = joint_precision * joint_mean - precision_list(:,:,m,i) * mean_list(:,:,m,i) + update_precision_list(:,:,m,i) * update_mean_list(:,:,m,i);
            joint_precision = joint_precision - precision_list(:,:,m,i) + update_precision_list(:,:,m,i);
            joint_mean = joint_precision \ joint_mean;
            
            diff = diff + mean(abs(joint{m}.mean(:,id) - joint_mean).^2);
            joint{m}.precision(:,:,id) = joint_precision;
            joint{m}.mean(:,id) = joint_mean;
        end
    end
    diff = diff / 3 / N;
    precision_list = update_precision_list;
    mean_list = update_mean_list;
    
    % train error
    
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
    [~,~,~,auctrain] = perfcurve(y,predicted_prob,1);
    
    % test
     M = size(test,1);
     mu = ones([rank, M]);
     var = ones([rank, rank, M]);
     for m = 1: 3
            ids = test(:,m);
            mu = mu .* joint{m}.mean(:,ids);
            var = var .* multinv(joint{m}.precision(:,:,ids));
     end
     smu = sum(mu)';
     svar = reshape(sum(sum(var, 2), 1), [M, 1]);
     predicted_prob = normcdf(smu./sqrt(1 + svar));
     [~,~,~,auctest] = perfcurve(test(:,4),predicted_prob,1);
    
    if iter > max_iter
        break;
    end
    
   disp(sprintf('iter:%d  diff:%.8f auctrain:%.4f auctest:%.4f',iter,diff, auctrain, auctest));
   
    iter = iter + 1;
%     last_auc = auc;

     
    
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
 [~,~,~,auc] = perfcurve(test(:,4),predicted_prob,1);

end