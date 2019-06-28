clear;
close all;

% load data

load('lp-binary.mat');
load('binary.mat');
load('b.mat');


rank_list = [3 5 8 10];

num_rank = 4;
num_fold = 5;
k = 1;
for r = 1 : num_rank
    % average
    for nfold = 1 : 5
        subplot(4, 5, k);
        k = k + 1;
        hold on;
        iter = cep.iter(r,nfold);
        time = cep.time_stat{r, nfold}(1:iter);
        auc = cep.test_auc_stat{r, nfold}(1:iter);
        plot(time, auc, 'r', 'LineWidth',3);
%         plot(1:iter, auc, 'r', 'LineWidth', 3);
        
        iter = vi.iter(r,nfold);
        time = vi.time_stat{r, nfold}(1:iter);
        auc = vi.test_auc_stat{r, nfold}(1:iter);
        plot(time, auc, 'b', 'LineWidth',3);
%         plot(1:iter, auc, 'b', 'LineWidth', 3);
        
        
        iter = lp.iter(r,nfold);
        time = lp.time_stat{r, nfold}(1:iter);
        auc = lp.test_auc_stat{r, nfold}(1:iter);
        plot(time, auc, 'g', 'LineWidth',3);
%         plot(1:iter, auc, 'b', 'LineWidth', 3);
        
        title(sprintf('rank: %d, nfold: %d', rank_list(r), nfold));
    
        set(gca, 'FontSize', 10);
    %     ylim([0.1, 10^5]);
%         xlim([0,1]);
        legend('CEP', 'VI', 'LP'); 
        xlabel('Time', 'FontSize', 10);
        ylabel('AUC', 'FontSize', 10);
        hold off;
    end
end