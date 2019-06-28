clear;
close all;

% load data

load('lp-binary-performance.mat');
load('cep-binary-500.mat');
load('vi-binary-500.mat');


rank_list = [3 5 8 10];

num_rank = 4;
num_fold = 5;
k = 1;
for r = 1 : num_rank
    cep_auc = zeros(500,1);
    vi_auc = zeros(500,1);
    lp_auc = zeros(500,1);
    % average
    for nfold = 1 : num_fold
%         subplot(4, 5, k);
%         k = k + 1;
%         hold on;
        iter = cep.iter(r,nfold);
%         time = cep.time_stat{r, nfold}(1:iter);
        auc = cep.test_auc_stat{r, nfold}(1:iter);
        cep_auc = cep_auc + auc;
%         plot(time, auc, 'r', 'LineWidth',3);
%         plot(1:iter, auc, 'r', 'LineWidth', 3);
        
        iter = vi.iter(r,nfold);
%         time = vi.time_stat{r, nfold}(1:iter);
        auc = vi.test_auc_stat{r, nfold}(1:iter);
        vi_auc = vi_auc + auc;
%         plot(time, auc, 'b', 'LineWidth',3);
%         plot(1:iter, auc, 'b', 'LineWidth', 3);
        
        
        iter = lp.iter(r,nfold);
%         time = lp.time_stat{r, nfold}(1:iter);
        auc = lp.test_auc_stat{r, nfold}(1:iter);
        extended = zeros(500,1);
        extended(1:iter) = auc;
        extended(iter+1:end) = auc(end);
        lp_auc = lp_auc + extended;
%         plot(time, auc, 'g', 'LineWidth',3);
%         plot(1:500, extended, 'g', 'LineWidth', 3);
        
%         title(sprintf('rank: %d, nfold: %d', rank_list(r), nfold));
%     
%         set(gca, 'FontSize', 10);
%     %     ylim([0.1, 10^5]);
% %         xlim([0,1]);
%         legend('CEP', 'VI', 'LP'); 
%         xlabel('Iteration', 'FontSize', 10);
%         ylabel('AUC', 'FontSize', 10);
%         hold off;
    end
    cep_auc = cep_auc / num_fold;
    vi_auc = vi_auc / num_fold;
    lp_auc = lp_auc / num_fold;
    
    subplot(2, 2, r);
    hold on;
        plot(1:500, cep_auc, 'red', 'LineWidth', 3);
        plot(1:500, vi_auc, 'blue', 'LineWidth', 3);
        plot(1:500, lp_auc, 'green', 'LineWidth', 3);
        
        title(sprintf('rank: %d', rank_list(r)));
    
        set(gca, 'FontSize', 10);
    %     ylim([0.1, 10^5]);
%         xlim([0,1]);
        legend('CEP', 'VI', 'LP'); 
        xlabel('Iteration', 'FontSize', 10);
        ylabel('AUC', 'FontSize', 10);
    hold off;
end