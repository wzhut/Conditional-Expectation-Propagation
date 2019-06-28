rng(0);
rank_list = [3 5 8 10];
n = 5;
% continuous

aucs = zeros(4, 5);

time = zeros(4, 5);
dim = [10000,200,10000]; 
cfg.tol = 1e-5;
cfg.rho = 0.5;
cfg.max_iter = 500;
cfg.verbose = 1;

train = load('../tensor-data-large/dblp/dblp-large-tensor.txt');
test = load('../tensor-data-large/dblp/dblp.mat');
test = test.data.test;
train(:,1:3) = train(:,1:3) + 1;
dummy_test = [test{1}.subs test{1}.Ymiss];


for nfold = 1 : n
    train = train(randperm(size(train,1)),:);
    for r = 1: length(rank_list)
        rank = rank_list(r);
        disp(sprintf('nfold: %d rank: %d', nfold, rank)); 
        tic;
        [auc, iter, aucs, joint] = b_cpcep(train, dummy_test, rank, dim, cfg);
        time(r, nfold) = toc;
        
        avg_auc = 0;
        for i = 1 : 50
            t = [test{i}.subs test{i}.Ymiss];
            % % test
            M = size(t,1);
            mu = ones([rank, M]);
            var = ones([rank, rank, M]);
            for m = 1: 3
                ids = t(:,m);
                mu = mu .* joint{m}.mean(:,ids);
                var = var .* for_multinv(joint{m}.precision(:,:,ids));
            end
            smu = sum(mu)';
            svar = reshape(sum(sum(var, 2), 1), [M, 1]);
            predicted_prob = normcdf(smu./sqrt(1 + svar));
            [~,~,~,auc] = perfcurve(t(:,4),predicted_prob,1);
            avg_auc = avg_auc + auc;
        end
        avg_auc = avg_auc / 50;
        aucs(r, nfold) = avg_auc;
    end
end

save('dblp.mat');

