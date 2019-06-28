function [joint, alpha, beta] = initialize(train, dim, rank)

N = length(train);
% summary

v = 1;

joint = cell([3,1]);
prior_mean = cell([3,1]);
for m = 1: 3
%     prior_mean{m} = mvnrnd(normrnd(0,1,[rank, 1]), eye(rank) * v, dim(m))';
    prior_mean{m} = mvnrnd(zeros([rank, 1]), eye(rank) * v, dim(m))';
    joint{m}.mean = zeros([rank, dim(m)]);
    joint{m}.precision = zeros([rank, rank, dim(m)]);
    for i = 1 : dim(m)
        num = sum(double(train(:,m) == i));
        precision = num * eye(rank) * 1e-6 + v * eye(rank);
        mean = precision \ (v * prior_mean{m}(:,i));
        joint{m}.mean(:,i) = mean;
        joint{m}.precision(:,:,i) = precision;
    end
end

alpha_0 = 0.01;
beta_0 = 0.01;

alpha = alpha_0;
beta = beta_0 + 0.01 * N;

end