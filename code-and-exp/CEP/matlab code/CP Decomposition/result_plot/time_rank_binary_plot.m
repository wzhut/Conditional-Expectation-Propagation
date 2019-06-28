clear;
close all;

% load data

load('lp-binary-5.mat');
load('cep-binary-500.mat');
load('vi-binary-500.mat');


rank_list = [3 5 8 10];

num_rank = 4;
num_fold = 5;
k = 1;
ceptime = zeros(num_rank,1);
lptime = zeros(num_rank, 1);
vitime = zeros(num_rank, 1);
for r = 1 : num_rank   
    for nfold = 1 : num_fold
        ceptime(r) = ceptime(r) + cep.time_stat{r,nfold}(end) / cep.iter(r, nfold);
        lptime(r) = lptime(r) + lp.time_stat{r, nfold}(end) / lp.iter(r, nfold);
        vitime(r) = vitime(r) + vi.time_stat{r, nfold}(end) / vi.iter(r, nfold);
    end  
end
ceptime = ceptime / num_fold;
lptime = lptime / num_fold;
vitime = vitime /num_fold;
hold on;
 plot(rank_list ,ceptime, 'redo--', 'LineWidth',3);
 plot(rank_list ,vitime, 'blueo--', 'LineWidth',3);
 plot(rank_list ,lptime, 'greeno--', 'LineWidth',3);

%  title(sprintf('rank: %d', rank_list(r), nfold));
% set(gca, 'YScale', 'log');
 set(gca, 'FontSize', 10);
%     ylim([0.1, 10^5]);
        xlim([3,10]);
 legend('CEP', 'VI', 'LP'); 
 xlabel('Rank', 'FontSize', 10);
 ylabel('Time', 'FontSize', 10);
hold off;