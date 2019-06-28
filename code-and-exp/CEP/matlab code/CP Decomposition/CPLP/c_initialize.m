function [joint] = c_initialize(train, rank, dim)

N = size(train, 1);
M = length(dim);

joint.mean = cell(M, 1);
joint.precision = cell(M, 1);
joint.alpha = 0;
joint.beta = 0;

for m = 1 : M
%     prior_mean = mvnrnd(zeros(rank, 1), eye(rank), dim(m))';
    prior_mean = rand([rank, dim(m)]);
    joint.mean{m} = zeros(rank, dim(m));
    joint.precision{m} = zeros(rank, rank, dim(m));
    for i = 1 : dim(m)
        num = sum(double(train(:,m) == i));
        precision = num * eye(rank) * 1e-6 + eye(rank);
        mean = precision \ prior_mean(:, i);
        joint.mean{m}(:,i) = mean;
        joint.precision{m}(:,:,i) = precision;
    end
end

joint.alpha = 1;
joint.beta = 1 + N * 0.01;

end