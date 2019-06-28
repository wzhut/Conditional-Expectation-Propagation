function [ mse, rmse ] = cpep_e( train, test, rank, dim)




mean_list = zeros([rank, N, 3]);
precision_list = reshape(repmat(reshape(eye(rank) * 1e-6, [rank*rank, 1]), [N * 3, 1]) , [rank, rank, N, 3]);
alpha_list = repmat(0.01, [N 1]);
beta_list = repmat(0.01, [N 1]);

alpha_0 = 0.01;
beta_0 = 0.01;

pho = 0.5;
N = length(train);

threshold = 0.01;
% train
while true
    for i = 1 : N
        for m = 1 : 3
            index = (train(:,m) == train(i,m));
            
        end
        
    end
    % train error
end


end