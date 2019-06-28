clear;
close all;

% load data

load('cep-continuous-500.mat');
load('vi-continuous-500.mat');
load('lp-continuous-performance.mat');

rank_list = [3 5 8 10];

num_rank = 4;
num_fold = 5;
k = 1;
for r = 1 : num_rank
    cep_rmse = zeros(500 , 1);
    vi_rmse = zeros(500, 1);
    lp_rmse = zeros(500, 1);
    % average
    for nfold = 1 : num_fold
%         subplot(4, 5, k);
%         k = k + 1;
%         hold on;
        iter = cep.iter(r,nfold);
%         time = cep.time_stat{r, nfold}(1:iter);
        rmse = cep.rmse_stat{r, nfold}(1:iter);
        cep_rmse = cep_rmse + rmse;
%         plot(time, auc, 'r', 'LineWidth',3);
%         plot(1:iter, rmse, 'r', 'LineWidth', 3);
        
        iter = vi.iter(r,nfold);
%         time = vi.time_stat{r, nfold}(1:iter);
        rmse = vi.rmse_stat{r, nfold}(1:iter);
%         plot(time, auc, 'b', 'LineWidth',3);
%         plot(1:iter, rmse, 'b', 'LineWidth', 3);
        vi_rmse = vi_rmse + rmse;
        
        iter = lp.iter(r,nfold);
        rmse = lp.rmse_stat{r, nfold}(1:iter);
        extended = zeros(500,1);
        extended(1:iter) = rmse;
        extended(iter+1:end) = rmse(end);
        lp_rmse = lp_rmse + extended;
        
%         title(sprintf('rank: %d, nfold: %d', rank_list(r), nfold));
%     
%         set(gca, 'FontSize', 10);
%         ylim([0.75, 1]);
% %         xlim([0,1]);
%         legend('CEP', 'VI'); 
%         xlabel('Iteration', 'FontSize', 10);
%         ylabel('RMSE', 'FontSize', 10);
%         hold off;
    end
    cep_rmse = cep_rmse / num_fold;
    vi_rmse = vi_rmse / num_fold;
    lp_rmse = lp_rmse /num_fold;
    subplot(2,2,r);
    hold on;
    plot(1:500, cep_rmse, 'r', 'LineWidth',3);
    plot(1:500, vi_rmse, 'b', 'LineWidth',3);
    plot(1:500, lp_rmse, 'g', 'LineWidth',3);

    title(sprintf('rank: %d', rank_list(r)));

    set(gca, 'FontSize', 10);
%     ylim([0.75, 1]);
    %         xlim([0,1]);
    legend('CEP', 'VI', 'LP'); 
    xlabel('Iteration', 'FontSize', 10);
    ylabel('RMSE', 'FontSize', 10);
    hold off;
end